import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,package,sha

def main():
    receipt=ROOT/'results/E053_EVALUATE_completed.json';r=json.loads(receipt.read_text())['result'];selected=r['selected'];parts=selected.split('_')
    cfg=dict(experiment='E054',selected=selected,division_threshold=int(parts[0][1:])/100,radius=int(parts[1][1:])/10,visual='visual' in parts,mode=parts[-1] if parts[-1] in ('dense','weighted') else 'none',selection_receipt_sha256=sha(receipt),local_score=r['metrics'][selected]['score'],local_scope=r['scope'],dense_manifest_sha256=json.loads((ROOT/'results/E041_TRAIN_completed.json').read_text())['manifest_sha256'],static_manifest_sha256=json.loads((ROOT/'results/E039_TRAIN_completed.json').read_text())['manifest_sha256'])
    assert cfg['local_score']>.9502832669356378+1e-8,'No new validated candidate; review before building'
    (ROOT/'baseline/e054_submission.json').write_text(json.dumps(cfg,indent=2)+'\n')
    files=json.loads((ROOT/'kaggle/e050_submission/payload.json').read_text())['files']
    files+=[str(p).replace('\\','/') for p in Path('src/biohub_lab').glob('*.py') if p.name in ['public_postprocess.py','visual_capture.py','visual_candidates.py','visual_assignment.py','association_rank.py','detection_identity.py']]
    files+=['baseline/e054_submission.json','scripts/joint_submission_runner.py','scripts/joint_submission_finish.py','scripts/visual_submission_postprocess.py']
    kernels=[] if cfg['mode']=='none' else ['jarturo/biohub-e041-train']+(['jarturo/biohub-e039-train'] if cfg['mode']=='weighted' else [])
    folder=ROOT/'kaggle/e054_submission';assert not (ROOT/'results/E054_SUBMISSION_launch.json').exists()
    checks=package(Path('kaggle/event_graph_control'),folder,files,'scripts/event_control_runner.py','scripts/joint_submission_runner.py','biohub-e054-joint-complement','Biohub E054 Joint Complement',True,kernels);checks.pop('calibration_videos',None)
    nb=json.loads((folder/'notebook.ipynb').read_text());nb['cells'][0]['source']=['# Joint complementary tracking\nFull hidden-test-capable inference with the frozen E053 selection. Public repair configuration, optional visual assignment and dense/static weighted completion. No cached test predictions or annotation reads.']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n');checks['notebook_sha256']=sha(folder/'notebook.ipynb');(ROOT/'results/E054_SUBMISSION_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(cfg)
if __name__=='__main__':main()
