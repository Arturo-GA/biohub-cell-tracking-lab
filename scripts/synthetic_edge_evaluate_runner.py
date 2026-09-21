"""Official CPU evaluation against previously frozen two-model controls."""
import json,sys
from pathlib import Path
from visual_validation_runner import main as evaluate

def main(package):
    evaluate(package)
    path=Path('/kaggle/working/visual_validation/result.json');result=json.loads(path.read_text())
    controls={'harmonic':.9436828898822122,'visual':.9502832669356378}
    result.update(experiment='E056',previous_controls=controls,
        deltas_vs_previous={k:v['score']-controls[k] for k,v in result['summaries'].items()},
        improved_over_previous_best=max(v['score'] for v in result['summaries'].values())>controls['visual']+1e-8,
        scope='Same eight reused calibration videos. Public checkpoint real-image pretraining membership unknown; exploratory, not independent validation.')
    # The inherited E023 promotion compares the two NEW arms; never mistake it for a comparison to our existing best.
    result['legacy_within_run_promotion']=result.pop('promote');result['promote']=result['improved_over_previous_best']
    path.write_text(json.dumps(result,indent=2)+'\n');print('ENSEMBLE_SYNTH_RESULT',result,flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
