"""Add an independent fragmentation-rescue ablation before metric evaluation."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from independent_linker import load_independent_model,sample_frame,pair_logits,annotate_joins
from independent_rules import join_candidates

W=Path(__file__).parent;ROOT=W.parents[1]
SOURCE=ROOT/'outputs/e069/image_fast/e069_image_cache';OUTPUT=ROOT/'outputs/e069/local_cache'
model,sampler,sha=load_independent_model((W/'hengck_model_v12.py').read_text(),ROOT/'outputs/e069/weights/00000008.pth')
torch.set_num_threads(2)
panel=json.loads((SOURCE/'panel.json').read_text());summary=[]
for i,entry in enumerate(sorted(panel,key=lambda r:r['stem']),1):
    stem=entry['stem'];start=time.monotonic();path=OUTPUT/(stem+'.json.gz')
    with gzip.open(path,'rt') as f:record=json.load(f)
    if record.get('joins_scored'):continue
    with np.load(SOURCE/'bundles'/(stem+'.npz'),allow_pickle=False) as b:
        ids=b['nid'];raw=b['nodes'];frames=b['frames'];images=b['images'];lo,hi=map(float,b['quantiles']);pairs=b['edges']
    ns={int(k):dict(t=int(v[0]),z=float(v[1]),y=float(v[2]),x=float(v[3])) for k,v in zip(ids,raw)}
    es=[dict(source_id=int(s),target_id=int(d)) for s,d in pairs]
    cs=join_candidates(ns,es);available=set(map(int,frames));by_t=collections.defaultdict(list)
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
        l=pair_logits(model,cache[t],cache[t+1],(64,64,64));scored.extend(annotate_joins(entries,l,frame_ids[t],frame_ids[t+1]))
        cache={f:v for f,v in cache.items() if f==t+1}
    record.update(joins=scored,joins_scored=True,joins_missing_image_frames=len(cs)-len(scored))
    tmp=path.with_suffix('.partial')
    with gzip.open(tmp,'wt') as f:json.dump(record,f)
    tmp.replace(path)
    row=dict(stem=stem,candidates=len(cs),scored=len(scored),eligible=sum(c['mutual'] and c['row_margin']>=2 and c['col_margin']>=2 for c in scored),seconds=time.monotonic()-start)
    summary.append(row);print('LOCAL JOINS',i,'/32',json.dumps(row),flush=True)
(ROOT/'results/E069/local_join_inference.json').write_text(json.dumps(summary,indent=2)+'\n')
manifest=json.loads((OUTPUT/'e069_local_manifest.json').read_text())
manifest.update(protocol='E069 swaps plus independent fragment joins',
    implementation_sha256={f:hashlib.sha256((W/f).read_bytes()).hexdigest() for f in ('independent_rules.py','independent_linker.py','run_e069_local_joins.py')})
(OUTPUT/'e069_local_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
