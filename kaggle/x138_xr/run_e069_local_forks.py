"""Test independent appearance evidence against c3 heuristic divisions."""
import collections,gzip,json,time
from pathlib import Path
import numpy as np
import torch
from independent_linker import load_independent_model,sample_frame,pair_logits
from independent_rules import fork_veto_candidates
W=Path(__file__).parent;ROOT=W.parents[1]
SOURCE=ROOT/'outputs/e069/image_fast/e069_image_cache';OUTPUT=ROOT/'outputs/e069/local_cache'
model,sampler,sha=load_independent_model((W/'hengck_model_v12.py').read_text(),ROOT/'outputs/e069/weights/00000008.pth')
torch.set_num_threads(2);summary=[]
for i,entry in enumerate(sorted(json.loads((SOURCE/'panel.json').read_text()),key=lambda r:r['stem']),1):
    stem=entry['stem'];start=time.monotonic();path=OUTPUT/(stem+'.json.gz')
    with gzip.open(path,'rt') as f:record=json.load(f)
    with np.load(SOURCE/'bundles'/(stem+'.npz'),allow_pickle=False) as b:
        ids=b['nid'];raw=b['nodes'];frames=b['frames'];images=b['images'];lo,hi=map(float,b['quantiles']);pairs=b['edges'];prob=b['eprob']
    ns={int(k):dict(t=int(v[0]),z=float(v[1]),y=float(v[2]),x=float(v[3])) for k,v in zip(ids,raw)}
    es=[dict(source_id=int(s),target_id=int(d),edge_prob=float(p) if np.isfinite(p) else None) for (s,d),p in zip(pairs,prob)]
    cs=fork_veto_candidates(ns,es);available=set(map(int,frames));by_t=collections.defaultdict(list)
    for c in cs:
        if c['t'] in available and c['t']+1 in available:by_t[c['t']].append(c)
    frame_ids=collections.defaultdict(list)
    for k in sorted(ns):frame_ids[ns[k]['t']].append(k)
    volumes={int(t):j for j,t in enumerate(frames)};cache={};scored=[]
    for t,entries in sorted(by_t.items()):
        for f in (t,t+1):
            if f not in cache:
                v=np.maximum((images[volumes[f]].astype(np.float32)-lo)/(hi-lo+1e-6),0).astype(np.float32)
                cache[f]=sample_frame(model,sampler,v,frame_ids[f],ns)
        l=pair_logits(model,cache[t],cache[t+1],(64,64,64));ri={k:j for j,k in enumerate(frame_ids[t])};ci={k:j for j,k in enumerate(frame_ids[t+1])}
        for c in entries:
            s,d,k=ri[c['s']],ci[c['d']],ci[c['keep']]
            scored.append(dict(c,weak_logit=float(l[s,d]),keep_logit=float(l[s,k]),keep_top=bool(l[s].argmax()==k)))
        cache={f:v for f,v in cache.items() if f==t+1}
    record.update(forks=scored,forks_scored=True,forks_missing_image_frames=len(cs)-len(scored))
    tmp=path.with_suffix('.partial')
    with gzip.open(tmp,'wt') as f:json.dump(record,f)
    tmp.replace(path)
    row=dict(stem=stem,candidates=len(cs),scored=len(scored),eligible=sum(c['weak_logit']<-2 and c['keep_logit']>2 and c['keep_top'] for c in scored),seconds=time.monotonic()-start)
    summary.append(row);print('LOCAL FORKS',i,'/32',json.dumps(row),flush=True)
(ROOT/'results/E069/local_fork_inference.json').write_text(json.dumps(summary,indent=2)+'\n')
