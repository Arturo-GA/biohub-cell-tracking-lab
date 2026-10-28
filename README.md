# Biohub Cell Tracking Lab

Research code for [Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) (Kaggle Research Code Competition, September 2026): detect and track zebrafish nuclei through 3D space and time in light-sheet microscopy, including cell divisions.

**Final result: 1,066th of 3,947 teams** (top 27 %). Private score **0.917** on the selected pair; public leaderboard peak **0.957** (rank 231 on the public board). An unselected submission with our own trained division heads scored **0.923** private, the best the team produced.

## The problem

Each video is a 3D volume per time point. The metric scores a lineage graph: every nucleus must be detected (within a tolerance in microns), linked to itself in the next frame, and when a cell divides both daughters must be linked to the mother. Missing or spurious divisions are penalised heavily. Train has 199 annotated videos; the test set is hidden and inference runs in a Kaggle kernel with a time limit.

## What this repo contributes

The public starting point was the community "Harmonic Fusion" pipeline (detector + nearest-neighbour association, score 0.946). Everything in `src/biohub_lab/` is this project's own work on top of it, with every experiment (E000–E071) run against the **official metric on held-out videos** before any submission:

| Line of work | Approach | Outcome |
|---|---|---|
| **Division heads** (E004–E007, E035–E037) | Five-frame image crops → CNN with attention over candidate parents and a symmetric division head; trained excluding one embryo at a time | Improved AP of divisions locally; the 5-seed ensemble on top of the x138 base gave the best private score (0.923) |
| **Dense / temporal detectors** (E008–E009, E031, E061) | CELLECT and two-Gaussian separation at native resolution; a detector fine-tuned on 40 real videos + synthetic replay | Recovered 512 extra centres at 3 µm but did not convert into score: the bottleneck was linking, not detection |
| **Joint lineage selection** (E010–E013) | Candidate centres + two-daughter trajectory hypotheses solved as a MILP with learned edge scores | Coverage of annotated divisions 18/33 → 33/33, but precision fell; kept as diagnostic |
| **HOCT adapter** (E003) | Port of the HOCT hierarchical linker with 3D watershed morphology | −0.047 vs control; false divisions ×12; rejected and documented |
| **Trajectory post-processing** (E064–E068) | Robust smoothing of full trajectories, minimum track length, division-cost sweeps in the ILP | +0.004 public; the final pair comes from this line |
| **Evaluation harness** | Official metric copied from the organisers' repo, per-embryo splits, receipts with SHA-256 of code/weights/CSV for every run | 167 unit tests across 66 modules, 148 generated notebooks, every submission traceable to a `results/E*/` receipt |

The negative results are kept on purpose: most of the compute went into ideas that looked right and measured worse, and the log says exactly why each one was dropped.

## Repository layout

```
src/biohub_lab/     detectors, association, division models, MILP solver, post-processing, submission export
src/biohub_official own copy of the organisers' metric (BSD-3)
src/biohub_cellect/ unmodified CELLECT model definition (GPL-2.0, see licenses/)
scripts/            build_* (generate Kaggle notebooks), *_runner.py (what runs inside the kernel), audit_*/verify_* (receipts)
kaggle/             148 generated kernels, one folder per experiment
baseline/           pinned public baseline, provenance.json with SHA-256 of every external asset
results/            JSON receipts per experiment (scores, hashes, timings, Kaggle submission status)
docs/               one report per experiment (Spanish); docs/HISTORY.es.md is the day-by-day log
tests/              unit tests (python -m unittest discover -s tests)
```

Data, weights, credentials and prediction CSVs are not in Git.

## Reproduce

Python 3.12, PyTorch (CUDA 12.1 wheel), NumPy, SciPy, scikit-image, Zarr.

```bash
pip install -r requirements-local.txt
export PYTHONPATH=src                       # PowerShell: $env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m unittest discover -s tests -v     # 167 tests, no data needed (torch + zarr required)
python scripts/build_notebooks.py           # regenerate the kernels in kaggle/
python -m kaggle kernels push -p kaggle/visual_submission
```

Inference needs a Kaggle GPU kernel with the competition data and the public model datasets attached (listed in `baseline/provenance.json`).

## Licensing and attribution

This repository has **no global license**: it combines code under MIT (NucVerse3D, HOCT), GPL-2.0 (CELLECT), BSD-3 (official metric) and Apache-2.0 (public notebooks). Each third-party file keeps its license in `licenses/` and is listed in [NOTICE.md](NOTICE.md). The code in `src/biohub_lab/`, `scripts/` and `tests/` was written for this project.
