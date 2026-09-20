import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from biohub_lab.joint_division import proposals,select
from biohub_lab.submission import COLUMNS
from biohub_lab.evaluate import evaluate_csv
from association_cpu_runner import locate,evaluation_dir

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(package):
    start=time.monotonic();root=Path('/kaggle/working/joint_division');root.mkdir(exist_ok=True);inputs=Path('/kaggle/input')
    config=json.loads((package/'baseline/e036_joint_division.json').read_text())
    gp=locate(inputs,'temporal_graph_inputs/result.json');dp=locate(inputs,'dense_detector_predictions/result.json')
    assert sha(gp)==config['graph_manifest_sha256'] and sha(dp)==config['dense_manifest_sha256']
    graphs=json.loads(gp.read_text());dense_manifest=json.loads(dp.read_text());prepared=[]
    for item in graphs['videos']:
        p=gp.parent/item['graph'];assert sha(p)==item['graph_sha256']
        with np.load(p) as g:ids,coords,original=g['ids'],g['coords'],g['edges']
        index={int(n):i for i,n in enumerate(ids)};edges=np.array([[index[int(a)],index[int(b)]] for a,b in original],np.int64).reshape(-1,2)
        dr=next(r for r in dense_manifest['files'] if r['video']==item['video'] and r['mode']=='batch_bn');p=dp.parent/dr['file'];assert sha(p)==dr['sha256']
        with np.load(p) as g:dense=g['coords']
        events,report=proposals(coords,edges,dense)
        prepared.append(dict(video=item['video'],ids=ids,coords=coords,edges=edges,dense=dense,events=events,report=report))
        print('JOINT_PROPOSALS',item['video'],report,flush=True)
    # All hypotheses are frozen before loading annotations or evaluating any arm.
    frozen=[dict(video=r['video'],report=r['report'],events=r['events']) for r in prepared]
    (root/'proposals.json').write_text(json.dumps(frozen))
    train=next(p/'train' for p in [inputs/'biohub-cell-tracking-during-development',inputs/'competitions/biohub-cell-tracking-during-development'] if (p/'train').exists())
    evaluation=evaluation_dir(train,[r['video'] for r in prepared],root/'data');metrics={};changes={};hashes={}
    for arm in ['control','parents','joint']:
        path=root/(arm+'.csv');counter=0;changes[arm]=[]
        with path.open('w',newline='') as out:
            writer=csv.writer(out);writer.writerow(COLUMNS)
            for r in prepared:
                coords,edges=r['coords'],r['edges'];ids=r['ids']
                if arm!='control':
                    coords,edges,report,chosen=select(coords,edges,r['dense'],r['events'],joint=arm=='joint')
                    ids=np.r_[ids,np.arange(int(max(ids))+1,int(max(ids))+1+len(coords)-len(ids))]
                    changes[arm].append(dict(video=r['video'],**report))
                    (root/(r['video']+'_'+arm+'_events.json')).write_text(json.dumps(chosen,indent=2))
                for n,c in zip(ids,coords):writer.writerow([counter,r['video'],'node',int(n),*map(int,c),-1,-1]);counter+=1
                for a,b in edges:writer.writerow([counter,r['video'],'edge',-1,-1,-1,-1,-1,int(ids[a]),int(ids[b])]);counter+=1
        hashes[arm]=sha(path);metrics[arm]=evaluate_csv(path,evaluation);(root/(arm+'_metrics.json')).write_text(json.dumps(metrics[arm],indent=2));print('JOINT_METRIC',arm,metrics[arm]['summary'],flush=True)
    assert abs(metrics['control']['summary']['score']-config['control_score'])<1e-10
    (root/'result.json').write_text(json.dumps(dict(status='complete',config=config,proposals=[dict(video=r['video'],**r['report']) for r in prepared],metrics={a:m['summary'] for a,m in metrics.items()},changes=changes,csv_sha256=hashes,seconds=time.monotonic()-start,gpu=False,training=False,annotations_used_for_proposals=False,leaderboard_submitted=False),indent=2,allow_nan=False))
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
