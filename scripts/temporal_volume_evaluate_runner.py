import hashlib,json,time
from pathlib import Path
import numpy as np

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def one(pattern):
    paths=list(Path('/kaggle/input').rglob(pattern));assert len(paths)==1;return paths[0]

def main(package):
    start=time.monotonic();root=Path('/kaggle/working/temporal_volume_evaluation');root.mkdir(exist_ok=True)
    source=one('temporal_volume_training/result.json');prepared=one('temporal_volume_data/result.json')
    training=json.loads(source.read_text());data=json.loads(prepared.read_text());config=json.loads((package/'baseline/e033_temporal_volume.json').read_text())
    assert training['status']=='complete' and training['config']==data['config']==config and training['input_manifest_sha256']==sha(prepared)
    reports=[];rng=np.random.default_rng(config['seed'])
    for fold in training['folds']:
        assert fold['steps']==config['steps'] and fold['first_conv_change_l2']>0
        assert all(not n.startswith(fold['heldout_embryo']) for n in fold['training_videos'])
        for item in fold['outputs']:
            assert item['video'].startswith(fold['heldout_embryo']);p=source.parent/item['file'];assert sha(p)==item['sha256']
            record=next(r for r in data['videos'] if r['video']==item['video']);q=prepared.parent/record['candidates'];assert sha(q)==record['candidates_sha256']
            with np.load(q,allow_pickle=False) as d:ix=d['indices'];distance=d['distance'];valid=d['valid'];labels=d['labels']
            with np.load(p,allow_pickle=False) as e:trained=e['trained'];random=e['random']
            def cosine(h):return (h[ix[:,0],None]*h[ix[:,1:]]).sum(-1)
            geo=-distance/5.;shuffled=trained[rng.permutation(len(trained))]
            scores=dict(nearest=geo,random=geo+cosine(random)/config['temperature'],trained=geo+cosine(trained)/config['temperature'],shuffled=geo+cosine(shuffled)/config['temperature'],visual_only=cosine(trained))
            correct={k:int((np.where(valid,s,-1e9).argmax(1)==labels).sum()) for k,s in scores.items()}
            reports.append(dict(video=item['video'],heldout_embryo=fold['heldout_embryo'],targets=len(labels),**correct))
    assert len(reports)==16
    totals={k:sum(r[k] for r in reports) for k in ['targets','nearest','random','trained','shuffled','visual_only']}
    folds={e:{k:sum(r[k] for r in reports if r['heldout_embryo']==e) for k in totals} for e in ['44b6','6bba']}
    promote=(totals['trained']-totals['nearest']>=.005*totals['targets'] and totals['trained']-totals['shuffled']>=.005*totals['targets'] and all(f['trained']>=f['nearest'] for f in folds.values()))
    result=dict(status='complete',seconds=time.monotonic()-start,config=config,training_manifest_sha256=sha(source),videos=reports,totals=totals,folds=folds,proceed_to_full_graph=promote,leaderboard_submitted=False,scope='Represented ambiguous-parent ranking on reused calibration cohort; independent encoder across embryos but upstream detector provenance unknown; not leaderboard or full graph metric')
    (root/'result.json').write_text(json.dumps(result,indent=2));print('VOLUME_EVALUATION_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
