"""Run only pretrained image inference on the laptop GPU; no annotation access."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import time
import numpy as np
import torch
from independent_linker import load_independent_model,sample_frame,pair_logits,annotate_candidates

W=Path(__file__).parent


def run(source,output,bundles=None):
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'run_manifest.json').read_text())
    expected=json.loads((W/'k_e069_images/expected_manifest.json').read_text())
    assert manifest==expected
    panel=json.loads((source/'panel.json').read_text())
    assert len(panel)==32 and json.loads((source/'complete.json').read_text())['videos']==32
    model,sampler,sha=load_independent_model((W/'hengck_model_v12.py').read_text(),W.parents[1]/'outputs/e069/weights/00000008.pth')
    assert sha==expected['checkpoint_sha256']
    torch.set_num_threads(2)
    (output/'e069_local_manifest.json').write_text(json.dumps(dict(image_manifest=manifest,checkpoint_sha256=sha,device=torch.cuda.get_device_name(0)),indent=2))
    (output/'panel.json').write_text(json.dumps(panel,indent=2))
    completed=[];start=time.monotonic()
    bundles=bundles or source/'bundles'
    # Output downloads are alphabetical; overlap GPU work with the remaining
    # image download without changing the prespecified evaluation panel.
    for entry in sorted(panel,key=lambda r:r['stem']):
        stem=entry['stem'];before=time.monotonic();destination=output/(stem+'.json.gz')
        if destination.exists():
            with gzip.open(destination,'rt',encoding='utf-8') as f: previous=json.load(f)
            assert previous['stem']==stem and previous['checkpoint_sha256']==sha
            completed.append(stem);continue
        with gzip.open(source/'candidates'/(stem+'.json.gz'),'rt',encoding='utf-8') as f:record=json.load(f)
        # npz is lazy: deliberately load only image and detection arrays,
        # never truth_nodes/truth_edges/t_true during model inference.
        bundle_path=bundles/(stem+'.npz')
        deadline=time.monotonic()+1800
        while not bundle_path.exists() or bundle_path.stat().st_size==0:
            if time.monotonic()>deadline:raise TimeoutError(str(bundle_path))
            time.sleep(2)
        # The SDK writes one fully received file at a time. Retry only an
        # incomplete zip during that short write window, never a model error.
        import zipfile
        for attempt in range(10):
            try:
                with zipfile.ZipFile(bundle_path) as z:
                    if z.testzip() is not None:raise zipfile.BadZipFile('Incomplete bundle')
                break
            except zipfile.BadZipFile:
                if attempt==9:raise
                time.sleep(2)
        with np.load(bundle_path,allow_pickle=False) as bundle:
            ids=bundle['nid'];raw=bundle['nodes'];frames=bundle['frames'];images=bundle['images'];lo,hi=map(float,bundle['quantiles'])
        ns={int(k):dict(t=int(v[0]),z=float(v[1]),y=float(v[2]),x=float(v[3])) for k,v in zip(ids,raw)}
        frame_ids=collections.defaultdict(list)
        for k in sorted(ns):frame_ids[ns[k]['t']].append(k)
        volumes={int(t):i for i,t in enumerate(frames)}
        by_t=collections.defaultdict(list)
        for c in record['candidates']:by_t[c['t']].append(c)
        cache={};scored=[]
        for t,cs in sorted(by_t.items()):
            for frame in (t,t+1):
                if frame not in cache:
                    image=np.maximum((images[volumes[frame]].astype(np.float32)-lo)/(hi-lo+1e-6),0).astype(np.float32)
                    cache[frame]=sample_frame(model,sampler,image,frame_ids[frame],ns)
            logits=pair_logits(model,cache[t],cache[t+1],(64,64,64))
            scored.extend(annotate_candidates(cs,logits,frame_ids[t],frame_ids[t+1]))
            cache={f:v for f,v in cache.items() if f==t+1}
        assert len(scored)==len(record['candidates'])
        record.update(candidates=scored,seconds=time.monotonic()-before,checkpoint_sha256=sha)
        tmp=destination.with_suffix('.partial')
        with gzip.open(tmp,'wt',encoding='utf-8') as f:json.dump(record,f)
        tmp.replace(destination)
        completed.append(stem)
        print(f"LOCAL LINKER {len(completed)}/32 {stem}: {len(scored)} candidates, {record['seconds']:.1f}s",flush=True)
        del images,cache
    result=dict(videos=len(completed),stems=completed,seconds=time.monotonic()-start,device=torch.cuda.get_device_name(0),checkpoint_sha256=sha)
    (output/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--bundles',type=Path);a=p.parse_args();run(a.source,a.output,a.bundles)
