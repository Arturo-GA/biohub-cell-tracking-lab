"""Build a validated test submission using frozen temporal checkpoints."""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .evaluate import shapes_for
from .submission import COLUMNS,read_and_validate
from .temporal_inference import load_model,score_video_ensemble


CHECKPOINT_HASHES={
    '44b6':'d4f29dc44bac76750b52724c5821633ae449d84214bf597b329d79e49bb45456',
    '6bba':'f3a9241bbc09c212739a7baaa29487a72d5f53ffb38f17656fef912f12b344c6',
}


def predict_test(seed_csv,data_dir,model_dir,output_dir,expected_hashes=None):
    """Read only runtime test images; publish root CSV only after validation."""
    expected_hashes=CHECKPOINT_HASHES if expected_hashes is None else expected_hashes
    root=Path(output_dir); root.mkdir(parents=True,exist_ok=True)
    final=root/'submission.csv'
    if final.exists(): raise FileExistsError('Refuse to confuse an existing submission with this run')
    shapes=shapes_for(data_dir)
    if not shapes: raise ValueError('No runtime test videos')
    seeds=read_and_validate(seed_csv,shapes)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2 if device.type=='cuda' else 4)
    members=[]; provenance={}
    for group,expected in sorted(expected_hashes.items()):
        path=Path(model_dir)/group/'best.pt'
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha!=expected: raise ValueError(f'Checkpoint checksum mismatch: {group}')
        model,checkpoint=load_model(path,device); members.append((model,checkpoint))
        provenance[group]=dict(sha256=sha,step=checkpoint['step'],division_prior=checkpoint['division_prior'])
    start=time.monotonic(); stats={}; index=0
    pending=root/'temporal_predictions.pending.csv'
    with pending.open('w',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(COLUMNS)
        for name,(nodes,control_edges) in sorted(seeds.items()):
            node_ids=sorted(nodes); remap={old:new for new,old in enumerate(node_ids)}
            coords=np.array([[nodes[n][k] for k in ('t','z','y','x')] for n in node_ids],np.float32)
            began=time.monotonic()
            edges,stat,arrays=score_video_ensemble(members,coords,Path(data_dir)/(name+'.zarr'),device)
            current=set(map(tuple,edges)); control={(remap[s],remap[t]) for s,t in control_edges}
            stat.update(seconds=time.monotonic()-began,control_edges=len(control),
                edges_added_vs_control=len(current-control),edges_removed_vs_control=len(control-current))
            stats[name]=stat
            np.savez_compressed(root/(name+'_scores.npz'),coords=coords,selected=edges,**arrays)
            for n,(t,z,y,x) in enumerate(coords):
                writer.writerow([index,name,'node',n,int(t),int(z),int(y),int(x),-1,-1]); index+=1
            for s,t in edges:
                writer.writerow([index,name,'edge',-1,-1,-1,-1,-1,int(s),int(t)]); index+=1
            print('TEMPORAL_TEST',name,json.dumps({k:v for k,v in stat.items() if k!='solver'}),flush=True)
    validated=read_and_validate(pending,shapes)
    if sum(len(nodes) for nodes,_ in validated.values())!=sum(len(nodes) for nodes,_ in seeds.values()):
        raise ValueError('Temporal inference changed the detector node count')
    pending.replace(final)
    receipt=dict(experiment='E005',scope='test inference; no ground-truth evaluation',
        models=provenance,model_source='jarturo/biohub-lab-temporal-train version 2',
        ensemble_method='Equal parent probabilities; equal division probabilities after per-model training-prior correction; one joint solver.',
        selection='Both checkpoints used on every video; no dataset-prefix routing.',
        statistics=stats,shapes=shapes,seconds=time.monotonic()-start,
        nodes=sum(len(n) for n,_ in validated.values()),edges=sum(len(e) for _,e in validated.values()),
        detector_sha256=hashlib.sha256(Path(seed_csv).read_bytes()).hexdigest(),
        submission_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
        validated=True,leaderboard_submitted=False,quality_status='awaiting_leaderboard_probe')
    (root/'temporal_test_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('TEMPORAL_TEST_COMPLETE',json.dumps({k:receipt[k] for k in ('nodes','edges','seconds','submission_sha256')}),flush=True)
    return receipt
