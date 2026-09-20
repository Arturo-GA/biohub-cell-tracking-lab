"""Package one frozen CPU-only integration diagnostic, no parameter sweep."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,package,sha
def main():
    files=read(ROOT/'kaggle/anchored_tracklets/payload.json')['files']+['scripts/segmentation_tracklets_runner.py','src/biohub_lab/segmentation_tracklets.py','src/biohub_lab/nucverse_instances.py']
    folder=ROOT/'kaggle/segmentation_tracklets'
    result=package(Path('kaggle/identity_parent'),folder,files,'scripts/identity_parent_runner.py','scripts/segmentation_tracklets_runner.py','biohub-lab-segmentation-tracklets-cpu','Biohub Lab Segmentation Tracklets CPU',False,['jarturo/biohub-lab-full-calibration-compare-cpu','jarturo/biohub-lab-error-window-evaluation-cpu'])
    nb=read(folder/'notebook.ipynb');nb['cells'][0]['source']=['# E029 complementary segmentation tracklets\nPreserve every control node and edge. Add only new mutually nearest consecutive detections. CPU, integer final CSV and official graph metric. Donor windows were annotation-selected: diagnostic only, no independent validation or submission.\n'];(folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n');result['notebook_sha256']=sha(folder/'notebook.ipynb')
    (ROOT/'results/E029_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
