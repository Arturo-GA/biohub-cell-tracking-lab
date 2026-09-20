"""E034 separate CPU proposal, GPU feature, and CPU official evaluation stages."""
import csv, hashlib, json, time
from pathlib import Path
import numpy as np
from biohub_lab.temporal_bridge import propose, augment

def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def one(name):
    root = Path('/kaggle/input')
    # Only mount roots: never traverse every image chunk of the competition.
    parents = [root, *root.glob('*'), *root.glob('*/*'), *root.glob('*/*/*')]
    paths = list(dict.fromkeys(p/name for p in parents if (p/name).is_file()))
    assert len(paths) == 1, (name, paths)
    return paths[0]

def checked(manifest, item, key, hashkey):
    p = manifest.parent / item[key]
    assert sha(p) == item[hashkey]
    return p

def main(package):
    start = time.monotonic()
    config = json.loads((package/'baseline/e034_bridge.json').read_text())
    stage = config['stage']; root = Path('/kaggle/working/temporal_bridge_'+stage); root.mkdir(exist_ok=True)
    gp = one('temporal_graph_inputs/result.json')
    assert sha(gp) == config['graph_manifest_sha256']
    graphs = json.loads(gp.read_text()); records = []
    if stage == 'prepare':
        dp = one('dense_detector_predictions/result.json')
        assert sha(dp) == config['dense_manifest_sha256']
        dense_manifest = json.loads(dp.read_text())
        for item in graphs['videos']:
            with np.load(checked(gp,item,'graph','graph_sha256')) as g:
                ids, coords, edges = g['ids'], g['coords'], g['edges']
            dr = next(r for r in dense_manifest['files'] if r['video']==item['video'] and r['mode']=='batch_bn')
            with np.load(checked(dp,dr,'file','sha256')) as d: dense = d['coords']
            proposals = propose(ids,coords,edges,dense)
            selected = sorted({i for p in proposals for i in p['donors']})
            remap = {d:i for i,d in enumerate(selected)}
            for p in proposals: p['donors'] = [remap[d] for d in p['donors']]
            path = root/(item['video']+'_donors.npy'); np.save(path,dense[selected],allow_pickle=False)
            records.append(dict(video=item['video'],file=path.name,sha256=sha(path),proposals=proposals,donors=len(selected)))
            print('BRIDGE_PREPARE',item['video'],len(proposals),len(selected),flush=True)
    elif stage == 'features':
        import torch
        from biohub_lab.temporal_volume import TemporalVolumeEncoder, gather_temporal
        assert torch.cuda.is_available(); torch.set_num_threads(2)
        pp = one('temporal_bridge_prepare/result.json'); prep = json.loads(pp.read_text())
        assert prep['status']=='complete' and prep['config']['graph_manifest_sha256']==config['graph_manifest_sha256']
        tp = one('temporal_volume_training/result.json'); assert sha(tp)==config['training_manifest_sha256']
        models = {}
        for f in json.loads(tp.read_text())['folds']:
            w=checked(tp,f,'checkpoint','checkpoint_sha256'); saved=torch.load(w,map_location='cpu',weights_only=True)
            net=TemporalVolumeEncoder().cuda().eval(); net.load_state_dict(saved['state_dict'],strict=True); models[f['heldout_embryo']]=net
        for item in prep['videos']:
            if time.monotonic()-start>600: raise TimeoutError('GPU inference budget exceeded')
            coords=np.load(checked(pp,item,'file','sha256'),allow_pickle=False)
            gi=next(g for g in graphs['videos'] if g['video']==item['video'])
            parts=[]
            if len(coords):
                image=torch.from_numpy(np.load(checked(gp,gi,'image','image_sha256'),allow_pickle=False)).cuda()
                centers=torch.as_tensor(np.rint(coords/np.array([1,1,4,4])).astype(np.int64),device='cuda')
                with torch.inference_mode():
                    for first in range(0,len(coords),128):
                        with torch.autocast('cuda',dtype=torch.float16):
                            h=models[item['video'].split('_')[0]](gather_temporal(image,centers[first:first+128]).float())
                        parts.append(h.float().cpu().numpy())
                del image
            path=root/(item['video']+'_features.npy'); np.save(path,np.concatenate(parts) if parts else np.empty((0,64),np.float32),allow_pickle=False)
            records.append(dict(video=item['video'],file=path.name,sha256=sha(path),donor_sha256=item['sha256']))
            print('BRIDGE_FEATURES',item['video'],len(coords),flush=True)
    else:
        from biohub_lab.submission import COLUMNS
        from biohub_lab.evaluate import evaluate_csv
        from association_cpu_runner import evaluation_dir
        pp=one('temporal_bridge_prepare/result.json'); prep=json.loads(pp.read_text())
        fp=one('temporal_bridge_features/result.json'); features=json.loads(fp.read_text())
        hp=one('temporal_graph_features/result.json'); assert sha(hp)==config['harmonic_features_sha256']; hs=json.loads(hp.read_text())
        inputs=Path('/kaggle/input'); train=next(p/'train' for p in [inputs/'biohub-cell-tracking-during-development',inputs/'competitions/biohub-cell-tracking-during-development'] if (p/'train').exists())
        evaluation=evaluation_dir(train,[g['video'] for g in graphs['videos']],root/'data')
        metrics={}; changes={}
        for arm in ['control','geometry','image']:
            path=root/(arm+'.csv'); counter=0; changes[arm]=[]
            with path.open('w',newline='') as out:
                writer=csv.writer(out); writer.writerow(COLUMNS)
                for gi in graphs['videos']:
                    with np.load(checked(gp,gi,'graph','graph_sha256')) as g: ids,coords,edges=g['ids'],g['coords'],g['edges']
                    if arm!='control':
                        item=next(r for r in prep['videos'] if r['video']==gi['video']); dense=np.load(checked(pp,item,'file','sha256'))
                        fr=next(r for r in features['videos'] if r['video']==gi['video']); assert fr['donor_sha256']==item['sha256']
                        hr=next(r for r in hs['outputs'] if r['video']==gi['video']); assert hr['graph_sha256']==gi['graph_sha256']
                        h=np.load(checked(hp,hr,'file','sha256')); dh=np.load(checked(fp,fr,'file','sha256'))
                        ids,coords,edges,report=augment(ids,coords,edges,dense,item['proposals'],h if arm=='image' else None,dh if arm=='image' else None)
                        changes[arm].append(dict(video=gi['video'],**report))
                    for n,c in zip(ids,coords): writer.writerow([counter,gi['video'],'node',int(n),*map(int,c),-1,-1]); counter+=1
                    for a,b in edges: writer.writerow([counter,gi['video'],'edge',-1,-1,-1,-1,-1,int(a),int(b)]); counter+=1
            metrics[arm]=evaluate_csv(path,evaluation)
            (root/(arm+'_metrics.json')).write_text(json.dumps(metrics[arm],indent=2))
            print('BRIDGE_METRIC',arm,metrics[arm]['summary'],flush=True)
        assert abs(metrics['control']['summary']['score']-config['control_score'])<1e-10
        records=dict(metrics={k:v['summary'] for k,v in metrics.items()},changes=changes)
    result=dict(status='complete',config=config,seconds=time.monotonic()-start,annotations_used_for_proposals=False,leaderboard_submitted=False)
    if stage=='evaluate': result.update(records)
    else: result['videos']=records
    (root/'result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
