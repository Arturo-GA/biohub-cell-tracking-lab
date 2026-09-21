"""E048 public compatible DivNet veto of existing forks, CPU inference+metric."""
import csv,json,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from ensemble_evaluate_runner import one,sha
from biohub_lab.public_divnet import PublicDivNet,input_patch
from biohub_lab.submission import COLUMNS
from biohub_lab.evaluate import evaluate_csv
from association_cpu_runner import evaluation_dir

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/public_divnet_veto');out.mkdir(exist_ok=True);cfg=json.loads((package/'baseline/e048_protocol.json').read_text());torch.set_num_threads(2)
    gp=one('temporal_graph_inputs/result.json');rp=one('dense_detector_inputs/result.json');assert sha(gp)==cfg['graph_sha256'] and sha(rp)==cfg['raw_sha256'];graphs=json.loads(gp.read_text());raw=json.loads(rp.read_text())
    # Locate the exact model file by the attached dataset and verify its hash.
    wp=one('best_overall.pt');assert sha(wp)==cfg['checkpoint_sha256'];saved=torch.load(wp,map_location='cpu',weights_only=True);model=PublicDivNet().eval();model.load_state_dict(saved['model_state'],strict=True)
    arms=['control','frame_veto','crop_veto'];handles={a:(out/(a+'.csv')).open('w',newline='') for a in arms};writers={a:csv.writer(h) for a,h in handles.items()};counter={a:0 for a in arms};rows=[];reports=[]
    for w in writers.values():w.writerow(COLUMNS)
    try:
        for item in graphs['videos']:
            video=item['video'];r=next(v for v in raw['videos'] if v['video']==video);gpfile=gp.parent/item['graph'];ip=rp.parent/r['file'];assert sha(gpfile)==item['graph_sha256'] and sha(ip)==r['sha256']
            with np.load(gpfile) as z:ids,coords,edges=z['ids'],z['coords'],z['edges']
            image=np.load(ip,mmap_mode='r');index={int(n):i for i,n in enumerate(ids)};children=defaultdict(list)
            for a,b in edges:children[int(a)].append(int(b))
            parents=sorted(a for a,c in children.items() if len(c)==2);pred={}
            for mode in ['frame','crop']:
                probs=[];cache={}
                for first in range(0,len(parents),8):
                    batch=np.stack([input_patch(image,coords[index[a]],mode,cache) for a in parents[first:first+8]])
                    with torch.inference_mode():probs.extend(torch.sigmoid(model(torch.from_numpy(batch))).numpy().tolist())
                pred[mode]=dict(zip(parents,probs))
            for a in parents:rows.append(dict(video=video,parent=a,**{m:pred[m][a] for m in pred}))
            for arm in arms:
                removed=set()
                if arm!='control':
                    mode=arm.split('_')[0]
                    for a,p in pred[mode].items():
                        if p<.5:
                            b=max(children[a],key=lambda b:(float(np.linalg.norm((coords[index[b],1:]-coords[index[a],1:])*[1.625,.40625,.40625])),b));removed.add((a,b))
                for n,c in zip(ids,coords):writers[arm].writerow([counter[arm],video,'node',int(n),*map(int,c),-1,-1]);counter[arm]+=1
                for a,b in edges:
                    if (int(a),int(b)) not in removed:writers[arm].writerow([counter[arm],video,'edge',-1,-1,-1,-1,-1,int(a),int(b)]);counter[arm]+=1
                reports.append(dict(video=video,arm=arm,original_divisions=len(parents),removed=len(removed)))
            print('ENSEMBLE_DIVNET',video,len(parents),flush=True)
    finally:
        for h in handles.values():h.close()
    inputs=Path('/kaggle/input');train=next(p/'train' for p in [inputs/'biohub-cell-tracking-during-development',inputs/'competitions/biohub-cell-tracking-during-development'] if (p/'train').exists());evaluation=evaluation_dir(train,[v['video'] for v in graphs['videos']],out/'data');metrics={};hashes={}
    for arm in arms:
        path=out/(arm+'.csv');hashes[arm]=sha(path);metric=evaluate_csv(path,evaluation);assert sha(path)==hashes[arm];metrics[arm]=dict(metric['summary'],**{k:sum(s[k] for s in metric['samples']) for k in ['edge_tp','edge_fp','edge_fn']});(out/(arm+'_metrics.json')).write_text(json.dumps(metric,indent=2));print('ENSEMBLE_DIVNET_METRIC',arm,metrics[arm],flush=True)
    c=metrics['control'];assert abs(c['score']-.9007529595605399)<1e-10
    passed={a:metrics[a]['score']>=c['score']+.001 and metrics[a]['division_tp']>=c['division_tp'] and metrics[a]['division_fp']<c['division_fp'] and metrics[a]['edge_tp']>=c['edge_tp'] and metrics[a]['edge_fp']<=c['edge_fp'] for a in arms if a!='control'}
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,metrics=metrics,passed=passed,rows=rows,reports=reports,csv_sha256=hashes,seconds=time.monotonic()-start,leaderboard_submitted=False,scope='Exploratory reused calibration; public best_overall training membership unknown (artifact reports199 movies); two prespecified normalization interpretations, not independent validation.'),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
