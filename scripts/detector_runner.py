"""E008/E009: raw-image proposal generation, full Harmonic tracking, official score."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from notebook_runner import competition_dir, diagnostic_data
from specialist_runner import source_root


def worker(method, data, root, names, weights):
    import torch
    torch.set_num_threads(2)
    if method == 'cellect':
        from biohub_lab.cellect_detector import run_video
    elif method == 'gaussian':
        from biohub_lab.gaussian_detector import run_video
    else:
        raise ValueError(method)
    for name in names:
        start = time.monotonic()
        args = (Path(data)/(name+'.zarr'), Path(root)/(name+'.npz'))
        result = run_video(*args, weights=weights) if method == 'cellect' else run_video(*args)
        print('PROPOSALS_READY',name,len(result['frame_counts']),result['proposals'],
              'seconds',time.monotonic()-start,flush=True)


def run_workers(method, data, root, names, weights, env):
    import torch
    count = min(2, torch.cuda.device_count())
    if count < 1:
        raise RuntimeError('GPU runtime required for full Harmonic experiment')
    processes = []
    try:
        for index in range(count):
            command = [sys.executable,'-u',__file__,'worker',method,str(data),str(root),
                       json.dumps(names[index::count]),str(weights)]
            processes.append(subprocess.Popen(command,env=dict(env,CUDA_VISIBLE_DEVICES=str(index))))
        while any(p.poll() is None for p in processes):
            failed = [p.returncode for p in processes if p.poll() not in (None,0)]
            if failed:
                raise RuntimeError(f'Proposal worker failed: {failed}')
            time.sleep(1)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('Proposal worker failed')
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                p.wait()


def main(method, package):
    from biohub_lab.patch import patch_source
    from biohub_lab.detector_proposals import install_pipeline_hook, CONFIG
    from biohub_lab.evaluate import evaluate_csv, shapes_for
    from biohub_lab.submission import read_and_validate
    package = Path(package).resolve()
    root = Path('/kaggle/working/detector_experiment')
    root.mkdir(exist_ok=True)
    proposals = root/'proposals'; proposals.mkdir(exist_ok=True)
    tracking = root/'tracking'; tracking.mkdir(exist_ok=True)
    data, names = diagnostic_data(competition_dir(),root)
    control_root = source_root('biohub-lab-official-metric-ab','run_receipt.json')
    previous = json.loads((control_root/'run_receipt.json').read_text())
    if set(names) != set(previous['datasets']):
        raise ValueError('Control and candidate datasets differ')
    control = previous['arms']['control']
    control_csv = control_root/'biohub_control/submission.csv'
    if hashlib.sha256(control_csv.read_bytes()).hexdigest() != control['sha256']:
        raise ValueError('Cached control digest mismatch')
    env = dict(os.environ, BIOHUB_LAB_SRC=str(package/'src'),
        BIOHUB_LAB_DATA_DIR=str(data), BIOHUB_PROPOSAL_DIR=str(proposals),
        PYTHONPATH=str(package/'src'), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='2')
    weights = ''
    if method == 'cellect':
        candidates = list(Path('/kaggle/input').rglob('U-ext+-x3rd-149.0-4.6540.pth'))
        if len(candidates) != 1:
            raise ValueError(f'Attach exactly one pinned CELLECT checkpoint: {candidates}')
        weights = str(candidates[0])
    receipt = dict(method=method,datasets=names,config=CONFIG,status='generating_proposals',
        scope='Conditional training diagnostic: all four videos occur in the public secondary detector training set. '
              'Prior error analysis used these four videos. This is not independent validation or a leaderboard result.',
        ground_truth_used_for_proposals=False, baseline_checkpoints_unchanged=True,
        full_harmonic_graph_and_postprocessing=True, leaderboard_submitted=False)
    receipt_path = root/'result.json'
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    start = time.monotonic()
    run_workers(method,data,proposals,names,weights,env)
    receipt.update(proposal_seconds=time.monotonic()-start,status='tracking',
        proposal_receipts={name:json.loads((proposals/(name+'.json')).read_text()) for name in names})
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    source = (package/'baseline/harmonic_inference.py').read_text()
    changed = install_pipeline_hook(patch_source(source,candidate=False))
    changed = changed.replace('/kaggle/working',str(tracking))
    compile(changed,'<detector-experiment>','exec')
    script = tracking/'run.py'; script.write_text(changed)
    track_start = time.monotonic()
    subprocess.run([sys.executable,'-u',str(script)],env=env,cwd=tracking,check=True)
    receipt['tracking_seconds'] = time.monotonic()-track_start
    # Only now access annotations through the independent official evaluator.
    final = tracking/'submission.csv'
    shapes = shapes_for(data)
    parsed = read_and_validate(final,shapes)
    metrics = evaluate_csv(final,data)
    metrics['scope'] = receipt['scope']
    injections = []
    for p in sorted(proposals.glob('injection_*.jsonl')):
        injections.extend(json.loads(line) for line in p.read_text().splitlines())
    expected = {(name,t) for name,shape in shapes.items() for t in range(shape[0])}
    actual = [(row['dataset'],row['t']) for row in injections]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('Missing or duplicate runtime injection frames')
    control_summary = control['metrics']['summary']
    receipt.update(status='complete',seconds=time.monotonic()-start,metrics=metrics,
        control=control_summary,delta={k:metrics['summary'][k]-control_summary[k]
            for k in ('score','adj_edge_jaccard','division_jaccard')},
        csv_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
        run_source_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        injection_summary={name:{key:sum(row[key] for row in injections if row['dataset']==name)
            for key in ('baseline','proposals','added')} for name in names},
        final_counts={name:dict(nodes=len(nodes),edges=len(edges)) for name,(nodes,edges) in parsed.items()},
        validated=True,quality_status='requires_comparison_review')
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print('DETECTOR_RESULT',json.dumps({k:v for k,v in receipt.items() if k!='proposal_receipts'},indent=2),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='worker':
        worker(sys.argv[2],sys.argv[3],sys.argv[4],json.loads(sys.argv[5]),sys.argv[6])
    else:
        main(sys.argv[1],sys.argv[2])
