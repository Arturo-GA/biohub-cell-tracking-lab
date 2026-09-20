"""Prepare benchmark images on CPU, so GPU performs neural inference only."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,package,sha
def main():
    names=read(ROOT/'baseline/event_graph_split.json')['split']['calibration']
    selected=sum([sorted(n for n in names if n.startswith(g))[:2] for g in ['44b6_','6bba_']],[])
    config=dict(videos=selected,frames=[25,75],selection='First two calibration names per specimen, fixed frame indices, no label-conditioned crops',
        preprocessing='Native voxel spacing; percentiles 2/99.8 without deconvolution; fixed generalist unscaled weights',
        patch=[64,128,128],stride=[48,96,96],gpu_deadline_seconds=1800,
        evaluation='One-to-one physical point coverage and complementary missed nuclei; sparse unmatched detections are not false-positive labels')
    (ROOT/'baseline/e024_benchmark.json').write_text(json.dumps(config,indent=2)+'\n')
    files=read(ROOT/'kaggle/visual_validation/payload.json')['files']+['scripts/nucverse_inputs_runner.py','baseline/e024_benchmark.json']
    folder=ROOT/'kaggle/nucverse_inputs'
    result=package(Path('kaggle/identity_parent'),folder,files,'scripts/identity_parent_runner.py','scripts/nucverse_inputs_runner.py','biohub-lab-nucverse-inputs-cpu','Biohub Lab NucVerse Inputs CPU',False,['jarturo/biohub-lab-harmonic-calibration-control'])
    nb=read(folder/'notebook.ipynb');nb['cells'][0]['source']=['# E024 fixed-frame CPU preparation\nEight full development frames for a new volumetric instance segmenter. No training or submission.\n']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    result.update(notebook_sha256=sha(folder/'notebook.ipynb'),calibration_videos=len(selected),frames=8);(ROOT/'results/E024_INPUTS_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
