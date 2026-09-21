"""Frozen complementary image inference followed by CPU weighted graph fusion."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
import torch,zarr
from ensemble_evaluate_runner import one,sha
from biohub_lab.dense_center import DenseCenter
from biohub_lab.temporal_detector import CenterField,vote_map
from biohub_lab.detector_complements import peaks,standardized
from biohub_lab.weighted_bridge import combine
from biohub_lab.calibration_export import export_control
from biohub_lab.submission import read_and_validate,COLUMNS

def main(package,test,control,config_name='e050_submission.json',graph_source=None,output_name='weighted_submission'):
    start=time.monotonic();out=Path('/kaggle/working')/output_name;cfg=json.loads((package/'baseline'/config_name).read_text())
    shapes={p.stem:tuple(zarr.open_group(str(p),mode='r')['0'].shape) for p in sorted(test.glob('*.zarr'))}
    validated=out/'public_tight55.csv';repair=export_control(graph_source or control/'submission.csv',validated,shapes);graphs=read_and_validate(validated,shapes)
    repo=control/'tracking_repo/src';assert repo.is_dir();sys.path.insert(0,str(repo))
    from biohub_tracking.io import open_dataset
    dp=one('dense_supervision_training/result.json');dm=json.loads(dp.read_text());assert sha(dp)==cfg['dense_manifest_sha256']
    wp=dp.parent/dm['checkpoint'];assert sha(wp)==dm['checkpoint_sha256']
    assert torch.cuda.is_available();torch.set_num_threads(2)
    dense=DenseCenter().cuda().eval();dense.load_state_dict(torch.load(wp,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    static=None
    if cfg['mode']!='dense':
        sp=one('temporal_detector_training/result.json');sm=json.loads(sp.read_text());assert sha(sp)==cfg['static_manifest_sha256']
        sr=next(r for r in sm['models'] if r['channels']==1);wp=sp.parent/sr['checkpoint'];assert sha(wp)==sr['sha256']
        static=CenterField(1).cuda().eval();static.load_state_dict(torch.load(wp,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    reports=[];index=0;target=Path('/kaggle/working/submission.csv');temporary=out/'candidate.partial.csv'
    with temporary.open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(COLUMNS)
        for video in sorted(shapes):
            path=test/(video+'.zarr');ds=open_dataset(path,normalize=False,load_image=False,require_tracks=False)
            low,high=float(ds.quantiles['0.001']),float(ds.quantiles['0.999']);image=zarr.open_group(str(path),mode='r')['0']
            donors=[];consensus=[]
            for t in range(shapes[video][0]):
                # Reproduce E031 float16 storage then E047 clipping exactly.
                raw=np.maximum((np.asarray(image[t,::1,::4,::4],np.float32)-low)/(high-low+1e-6),0).astype(np.float16)
                frame=np.clip(raw.astype(np.float32),0,1);x=torch.as_tensor(frame,device='cuda')[None,None]
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    logits=dense(x)[0].cpu().numpy().astype(np.float16).astype(np.float32)
                    field=static(x)[0].cpu().numpy().astype(np.float16).astype(np.float32) if static is not None else None
                points=peaks(logits,512);donors.append(np.c_[np.full(len(points),t),points*[1,4,4]])
                if field is not None:
                    score=standardized(logits)+standardized(np.log1p(vote_map(field,frame)))
                    cp=peaks(score,512);consensus.append(np.c_[np.full(len(cp),t),cp*[1,4,4]])
            dn=np.concatenate(donors).astype(np.int64);cn=np.concatenate(consensus).astype(np.int64) if consensus else dn
            nodes,edges=graphs[video];ids=np.array(sorted(nodes));coords=np.array([[nodes[int(n)][k] for k in ['t','z','y','x']] for n in ids]);edges=np.asarray(edges,dtype=np.int64).reshape(-1,2)
            ns,cs,es,report=combine(ids,coords,edges,dn,cn,cfg['mode'])
            for n,c in zip(ns,cs):writer.writerow([index,video,'node',int(n),*map(int,c),-1,-1]);index+=1
            for a,b in es:writer.writerow([index,video,'edge',-1,-1,-1,-1,-1,int(a),int(b)]);index+=1
            reports.append(dict(video=video,nodes=len(ns),edges=len(es),**report));print('ENSEMBLE_SUBMISSION_VIDEO',reports[-1],flush=True)
    read_and_validate(temporary,shapes);temporary.replace(target)
    result=dict(status='complete',config=cfg,reports=reports,shapes=shapes,csv_sha256=sha(target),public_control_sha256=sha(validated),export_repair=repair,seconds=time.monotonic()-start,annotations_read=False,cached_test_predictions=False,leaderboard_submitted=False)
    (out/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
