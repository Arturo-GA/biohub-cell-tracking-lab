"""CPU diagnostic of exact rescued/lost GT edges; never a deployable oracle."""
import json,sys,time
from pathlib import Path
import tracksdata as td
from residual_detector_io import one,sha
from residual_audit_runner import match
from association_cpu_runner import evaluation_dir
from biohub_lab.submission import read_and_validate
from biohub_lab.evaluate import shapes_for

def main(package):
    start=time.monotonic();cfg=json.loads((package/'baseline/e063_protocol.json').read_text());pins=json.loads((package/'baseline/e063_audit_pins.json').read_text())
    out=Path('/kaggle/working/selector_residual_audit');out.mkdir(exist_ok=True);p=one('selector_evaluation/result.json');assert sha(p)==pins['EVALUATE'];source=json.loads(p.read_text());hashes=json.loads((p.parent/'frozen_predictions.json').read_text())['graphs']
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists());data=evaluation_dir(train,cfg['evaluation'],out/'data');shapes=shapes_for(data)
    graphs={};metrics={}
    for arm in ('control','rule','learned','conditional'):
        path=p.parent/(arm+'.csv');assert sha(path)==hashes[arm];graphs[arm]=read_and_validate(path,shapes)
        metrics[arm]={r['dataset']:r for r in json.loads((p.parent/(arm+'_metrics.json')).read_text())['samples']}
    rows=[]
    for video in cfg['evaluation']:
        truth=td.graph.IndexedRXGraph.from_geff(str(data/(video+'.geff')))[0];correct={};done={}
        for arm in graphs:
            g=graphs[arm][video];same=next((a for a in done if g==done[a]),None)
            if same is None:correct[arm]=match(*g,truth,metrics[arm][video])[0]
            else:
                assert all(metrics[arm][video][k]==metrics[same][video][k] for k in ('edge_tp','edge_fp','edge_fn'))
                correct[arm]=correct[same]
            done[arm]=g
        for arm in ('rule','learned','conditional'):
            rows.append(dict(video=video,arm=arm,rescued=sorted(correct[arm]-correct['control']),lost=sorted(correct['control']-correct[arm]),novel_beyond_rule=sorted(correct[arm]-(correct['control']|correct['rule']))))
        print('ENSEMBLE_SELECTOR_AUDIT',video,[dict(arm=r['arm'],rescued=len(r['rescued']),lost=len(r['lost']),novel_beyond_rule=len(r['novel_beyond_rule'])) for r in rows[-3:]],flush=True)
    totals={a:{k:sum(len(r[k]) for r in rows if r['arm']==a) for k in ('rescued','lost','novel_beyond_rule')} for a in ('rule','learned','conditional')}
    for a in totals:assert totals[a]['rescued']-totals[a]['lost']==source['metrics'][a]['edge_tp']-source['metrics']['control']['edge_tp']
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,totals=totals,rows=rows,seconds=time.monotonic()-start,leaderboard_submitted=False,scope='Diagnostic using evaluation annotations after the frozen experiment. Exact GT edge sets, not counts alone. No oracle selection exported for inference.'),indent=2));print('ENSEMBLE_SELECTOR_AUDIT_TOTAL',totals,flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
