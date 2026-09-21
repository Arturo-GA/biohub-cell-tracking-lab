"""CPU component diagnosis: sparse geometric coverage, not detector precision."""
import json,sys,time
from pathlib import Path
import numpy as np
from biohub_lab.temporal_data import load_gt
from biohub_lab.submission import read_and_validate
from biohub_lab.residual_tracklets import chains
from ensemble_evaluate_runner import matched
from residual_detector_io import one,sha

def main(package):
    start=time.monotonic();cfg=json.loads((package/'baseline/e061_protocol.json').read_text());pins=json.loads((package/'baseline/e061_evaluate_pins.json').read_text())
    out=Path('/kaggle/working/residual_coverage');out.mkdir(exist_ok=True)
    p=one('residual_detector_predictions/result.json');assert sha(p)==pins['predictions_sha256'];manifest=json.loads(p.read_text())
    gp=one('temporal_graph_inputs/result.json');assert sha(gp)==pins['graphs_sha256'];old=json.loads(gp.read_text());controls={}
    for r in old['videos']:
        q=gp.parent/r['graph'];assert sha(q)==r['graph_sha256']
        with np.load(q) as z:controls[r['video']]=z['coords']
    v=one('visual_validation/result.json').parent
    assert sha(v/'visual.csv')==json.loads((v/'frozen_predictions.json').read_text())['csv_sha256']['visual']
    shapes={r['video']:r['shape'] for r in manifest['records'] if r['video'] not in controls}
    for video,(nodes,_) in read_and_validate(v/'visual.csv',shapes).items():controls[video]=np.array([[n[k] for k in ('t','z','y','x')] for n in nodes.values()])
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists());reports=[]
    for video in cfg['evaluation']:
        gt,_=load_gt(train/(video+'.geff'));baseline=controls[video];data={}
        for model in ('frozen','adapted','repeated'):
            r=next(r for r in manifest['records'] if r['video']==video and r['arm']==model);q=p.parent/r['file'];assert sha(q)==r['sha256']
            with np.load(q) as z:coords=z['coords'];prob=z['prob'];features=z['features'].astype(np.float32)
            persistent=sorted({k for path in chains(coords,features) for k in path})
            data[model]=(coords,prob,set(persistent))
        rows={}
        for model in data:
            for mode in ('full','baseline_budget','persistent'):
                rows[model,mode]=dict(video=video,model=model,mode=mode,truth=len(gt),matched3=0,matched7=0,rescued7=0,lost7=0,candidates=0)
        base3=base7=0
        for t in np.unique(gt[:,0]):
            truth=gt[gt[:,0]==t,1:]/[1,4,4];base=baseline[baseline[:,0]==t,1:]/[1,4,4]
            b3=matched(base,truth,3);b7=matched(base,truth,7);base3+=len(b3);base7+=len(b7)
            for model,(coords,prob,persistent) in data.items():
                idx=np.flatnonzero(coords[:,0]==t);ordered=idx[np.argsort(-prob[idx],kind='stable')]
                for mode,selection in [('full',idx),('baseline_budget',ordered[:len(base)]),('persistent',np.array([i for i in idx if i in persistent],int))]:
                    centers=coords[selection,1:]/[1,4,4];m3=matched(centers,truth,3);m7=matched(centers,truth,7);r=rows[model,mode]
                    r['matched3']+=len(m3);r['matched7']+=len(m7);r['rescued7']+=len(m7-b7);r['lost7']+=len(b7-m7);r['candidates']+=len(selection)
        for r in rows.values():r.update(baseline3=base3,baseline7=base7)
        reports.extend(rows.values());print('ENSEMBLE_RESIDUAL_COVERAGE',video,[(r['model'],r['mode'],r['matched7'],r['rescued7']) for r in rows.values()],flush=True)
    totals=[]
    for model in ('frozen','adapted','repeated'):
        for mode in ('full','baseline_budget','persistent'):
            subset=[r for r in reports if r['model']==model and r['mode']==mode]
            totals.append(dict(model=model,mode=mode,**{k:sum(r[k] for r in subset) for k in ('truth','matched3','matched7','rescued7','lost7','candidates','baseline3','baseline7')}))
    (out/'result.json').write_text(json.dumps(dict(status='complete',totals=totals,by_video=reports,seconds=time.monotonic()-start,
        scope='Sparse one-to-one geometric coverage at annotated frames; not precision or official graph score. All 24 reused videos. No threshold selection.'),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
