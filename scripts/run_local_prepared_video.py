"""Import one finished CPU shard and extract its remaining features on the laptop."""
import argparse
from pathlib import Path
import subprocess
import sys
from import_cpu_inputs import import_package

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu-package',required=True)
    args=parser.parse_args();package=Path(args.cpu_package).resolve()
    imported=import_package(package,ROOT/'outputs/local_images')
    command=[sys.executable,'-u',str(ROOT/'scripts/prepare_event_video.py'),
        '--root',str(ROOT/'outputs/local_event_graph'),'--video',imported['video'],'--image',imported['image'],
        '--primary',str(ROOT/'artifacts/e012_detector_check/primary/edge_predictor_best.pth'),
        '--secondary',str(ROOT/'artifacts/e012_detector_check/secondary/edge_predictor_best.pth'),
        '--cellect',str(ROOT/'artifacts/cellect_public_weights/U-ext+-x3rd-149.0-4.6540.pth'),
        '--module',str(ROOT/'outputs/e012_v1/harmonic_dag_experiment/image_detector.py'),
        '--support-repo',str(ROOT/'artifacts/e012_detector_check/repo'),
        '--cached-e012',str(ROOT/'outputs/e012_v1/harmonic_dag_experiment'),
        '--gaussian-cache',str(package),'--device','cuda']
    subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':main()
