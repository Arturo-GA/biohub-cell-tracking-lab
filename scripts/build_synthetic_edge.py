"""Freeze the reviewed, attributed public synthetic specialist hook."""
import hashlib,json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,package,sha
from build_three_lines import build

def main():
    source=ROOT/'artifacts/research_latest_20260920/andnyu__biohub-947-synthetic-edge/source.py'
    assert sha(source)=='4b5aa851bae4891fe2c24935190664cecad3d7072b6c8838588f65000f00da36'
    text=source.read_text(encoding='utf8');start=text.index('# ================= SYNTHETIC LOW-MARGIN EDGE TIEBREAK');end=text.index('# ================= END SYNTHETIC LOW-MARGIN EDGE TIEBREAK',start)
    hook='# Adapted from https://www.kaggle.com/code/andnyu/biohub-947-synthetic-edge\n'+text[start:end]
    compile(hook,'reviewed_synthetic_hook','exec');(ROOT/'baseline/e056_synthetic_hook.py').write_text(hook,encoding='utf8')
    assert json.loads((ROOT/'results/E056_checkpoint_audit.json').read_text())['strict_load']
    protocol=dict(experiment='E056',source='andnyu/biohub-947-synthetic-edge',source_sha256=sha(source),hook_sha256=sha(ROOT/'baseline/e056_synthetic_hook.py'),videos=json.loads((ROOT/'baseline/e023_validation.json').read_text())['videos'],max_weight=.25,public_margin_max=.12,synthetic_margin_gain=.02,original_detections_unchanged=True,scope='Eight reused videos; exploratory, no independent holdout',decision='Compare both synthetic ILP and synthetic visual assignment against original .9436828898822122 and visual .9502832669356378. Do not submit a candidate that fails to improve the previous best.')
    (ROOT/'baseline/e056_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    files=json.loads((ROOT/'kaggle/visual_validation_capture/payload.json').read_text())['files']+['scripts/synthetic_edge_capture_runner.py','baseline/e056_protocol.json','baseline/e056_synthetic_hook.py','src/biohub_lab/public_postprocess.py']
    folder=ROOT/'kaggle/e056_capture';assert not (ROOT/'results/E056_CAPTURE_launch.json').exists()
    checks=package(Path('kaggle/event_graph_control'),folder,files,'scripts/event_control_runner.py','scripts/synthetic_edge_capture_runner.py','biohub-e056-synthetic-specialist','Biohub E056 Synthetic Specialist',True,[])
    meta=json.loads((folder/'kernel-metadata.json').read_text());meta['dataset_sources'].append('bhpepper/biohub-synthetic-5fold-ensemble-v1');(folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    nb=json.loads((folder/'notebook.ipynb').read_text());nb['cells'][0]['source']=['# E056 synthetic specialist\nFrozen third model on low-margin links. Eight reused calibration videos, no annotations read. Official evaluation follows on CPU.'];(folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n');checks['notebook_sha256']=sha(folder/'notebook.ipynb');checks.pop('calibration_videos',None)
    (ROOT/'results/E056_CAPTURE_preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
    files=json.loads((ROOT/'kaggle/visual_validation/payload.json').read_text())['files']+['scripts/synthetic_edge_evaluate_runner.py']
    build('E056_EVALUATE','scripts/synthetic_edge_evaluate_runner.py',files,['jarturo/biohub-e056-synthetic-specialist'])
    print(checks)
if __name__=='__main__':main()
