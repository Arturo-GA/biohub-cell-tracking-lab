"""Measure CUDA memory on real candidate geometry, with synthetic supervision."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import psutil
import torch
from biohub_lab.event_data import candidates, neighborhoods, node_inputs, save_json
from biohub_lab.event_model import EventGraphNet
from biohub_lab.event_train import load_arrays, positions, loss_for


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cached',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=100);args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Local CUDA unavailable')
    torch.set_num_threads(2);torch.manual_seed(20260915);rng=np.random.default_rng(20260915)
    paths=sorted((Path(args.cached)/'videos').glob('*/combined/graph.npz'))
    largest=None;stress=None
    for path in paths:
        with np.load(path,allow_pickle=False) as saved:coords=saved['coords']
        if largest is None or len(coords)>largest[0]:largest=(len(coords),path)
        for frame in np.unique(coords[:,0]):
            count=int(((coords[:,0]>=frame-4)&(coords[:,0]<=frame+5)).sum())
            if stress is None or count>stress[0]:stress=(count,path,int(frame))
    count,path,frame=stress;graph=load_arrays(path);coords=graph['coords']
    ids=np.flatnonzero((coords[:,0]>=frame-4)&(coords[:,0]<=frame+5))
    remap=np.full(len(coords),-1,np.int64);remap[ids]=np.arange(len(ids))
    neighbors=neighborhoods(coords)[ids];local=remap[neighbors.clip(0)];local[neighbors<0]=-1
    visual=rng.normal(size=(len(coords),64)).astype(np.float16);inputs=node_inputs(graph,visual)
    edges,triples=candidates(graph,np.flatnonzero(coords[:,0]==frame))
    edges=edges[:1024];triples=triples[:512]
    quality_ids=np.arange(min(512,len(ids)))
    batch=dict(inputs=torch.tensor(inputs[ids],device='cuda'),positions=torch.tensor(positions(coords[ids]),device='cuda'),
        neighbors=torch.tensor(local,device='cuda'),targets=dict(edges=torch.tensor(remap[edges],device='cuda'),
        edge_y=torch.tensor(rng.integers(0,2,len(edges)).astype(np.float32),device='cuda'),
        triples=torch.tensor(remap[triples],device='cuda'),triple_y=torch.tensor(rng.integers(0,2,len(triples)).astype(np.float32),device='cuda'),
        quality_ids=torch.tensor(quality_ids,device='cuda'),quality=torch.tensor(rng.random(len(quality_ids)).astype(np.float32),device='cuda')))
    model=EventGraphNet().cuda();model.attention_chunk=256;model.checkpoint_activations=True
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4);scaler=torch.amp.GradScaler('cuda')
    torch.cuda.reset_peak_memory_stats();times=[];rss=[]
    for step in range(args.steps):
        torch.cuda.synchronize();start=time.monotonic();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            loss=loss_for(model,batch)
        if not torch.isfinite(loss):raise ValueError('Benchmark loss is not finite')
        scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        times.append(time.monotonic()-start);rss.append(psutil.Process().memory_info().rss)
        if (step+1)%10==0:print('LOCAL_CAPACITY_STEP',step+1,round(times[-1],3),flush=True)
    training=dict(steps=args.steps,window_nodes=count,video=path.parent.parent.name,frame=frame,
        median_step_seconds=float(np.median(times[5:] if len(times)>5 else times)),
        maximum_step_seconds=max(times),max_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        max_reserved_mib=torch.cuda.max_memory_reserved()/2**20,max_process_rss_mib=max(rss)/2**20)
    del batch,optimizer,scaler,loss,graph,inputs,visual,coords,neighbors,local
    model.eval();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    graph=load_arrays(largest[1]);visual=np.zeros((len(graph['coords']),64),np.float16)
    inputs=node_inputs(graph,visual);neighbors=neighborhoods(graph['coords']);start=time.monotonic()
    encoded=model.encode_streamed(inputs,positions(graph['coords']),neighbors,'cuda',chunk=256)
    torch.cuda.synchronize()
    inference=dict(video=largest[1].parent.parent.name,nodes=largest[0],seconds=time.monotonic()-start,
        output_finite=bool(torch.isfinite(encoded).all()),max_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        max_reserved_mib=torch.cuda.max_memory_reserved()/2**20,process_rss_mib=psutil.Process().memory_info().rss/2**20)
    result=dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),torch=str(torch.__version__),cuda_runtime=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),total_vram_mib=torch.cuda.get_device_properties(0).total_memory/2**20,
        scope='Capacity only: real E012 candidate geometry, synthetic features/targets, no GT labels and no reusable trained weights',
        training=training,full_video_streamed_encoder=inference,scientific_training_started=False)
    save_json(args.output,result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
