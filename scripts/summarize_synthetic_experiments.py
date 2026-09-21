"""Compare every evaluable count per video, not just rounded aggregate scores."""
import json
from pathlib import Path
from datetime import datetime,timezone

def read(path):return json.loads(Path(path).read_text())

def main():
    result=read('results/E057_EVALUATE_completed.json')['result'];assert result['status']=='complete'
    hashes=read('outputs/e057_evaluate/synthetic_fixed_graph/frozen_predictions.json');assert len(set(hashes.values()))==4
    counts=['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn']
    rows=[]
    for arm in ['harmonic','visual']:
        old_arm='control' if arm=='harmonic' else 'visual'
        old={s['dataset']:s for s in read(f'outputs/e055_evaluate/selective_repair/{old_arm}_metrics.json')['samples']}
        for s in read(f'outputs/e056_evaluate/visual_validation/{arm}_metrics.json')['samples']:
            rows.append(dict(experiment='E056',arm=arm,video=s['dataset'],count_deltas={k:s[k]-old[s['dataset']][k] for k in counts},node_delta=s['num_pred_nodes']-old[s['dataset']]['num_pred_nodes']))
    old={s['dataset']:s for s in read('outputs/e057_evaluate/synthetic_fixed_graph/visual_metrics.json')['samples']}
    for arm in ['specialist','balanced','uncertainty']:
        for s in read(f'outputs/e057_evaluate/synthetic_fixed_graph/{arm}_metrics.json')['samples']:
            rows.append(dict(experiment='E057',arm=arm,video=s['dataset'],count_deltas={k:s[k]-old[s['dataset']][k] for k in counts},node_delta=s['num_pred_nodes']-old[s['dataset']]['num_pred_nodes']))
    assert all(all(v==0 for v in r['count_deltas'].values()) for r in rows)
    decision=dict(status='complete',decision='do_not_promote',leaderboard_submitted=False,leaderboard_target_achieved=False,latest_confirmed_public_score=.946,
        frozen_csv_sha256=hashes,all_four_csvs_different=True,
        all_evaluable_counts_unchanged_per_video=True,paired=rows,
        interpretation='The new expert and weighted pools change predictions but do not improve evaluable edge/division counts in any of the eight reused videos. Counts alone do not prove the same individual links remain correct, nor that every changed link is unannotated.',
        next_research_hypothesis='Investigate residual errors and calibrate expert routing on separate fitting data; low raw confidence alone has not isolated the observed failures. This next study is not executed or claimed successful.')
    Path('results/E056_E057_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    path=Path('results/STATUS.json');status=read(path)
    status['E056'].update(status='COMPLETE',decision='rejected',best_local_score=.9502779211712876,delta_vs_previous_best=-.00000534576435018419,seconds_cpu=read('results/E056_EVALUATE_completed.json')['result']['seconds'],pending_jobs=False)
    status['E057'].update(status='COMPLETE',decision='all_new_arms_tie_existing_visual',score=.9502832669356378,seconds_cpu=result['seconds'],pending_jobs=False,leaderboard_submitted=False)
    status['updated_at_utc']=datetime.now(timezone.utc).isoformat();path.write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps({k:v for k,v in decision.items() if k!='paired'},indent=2))

if __name__=='__main__':main()
