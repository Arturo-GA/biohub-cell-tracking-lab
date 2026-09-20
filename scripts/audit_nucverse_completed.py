"""Verify the new detector benchmark's deployed source and aggregate receipts."""
import json
from audit_e017_completed import ROOT,read,sha,verify
def main():
    gpu=ROOT/'outputs/e024_gpu_recovery';cpu=ROOT/'outputs/e024_eval_recovery'
    a=verify(gpu,'kaggle/nucverse_inference','results/E024_GPU_launch.json','nucverse_inference_package')
    b=verify(cpu,'kaggle/nucverse_evaluation','results/E024_EVAL_launch.json','nucverse_evaluation_package')
    inference=read(gpu/'nucverse_fields/result.json');evaluation=read(cpu/'nucverse_evaluation/result.json')
    assert inference['status']==evaluation['status']=='complete'
    assert not inference['annotations_read'] and not inference['training']
    frozen=read(cpu/'nucverse_evaluation/frozen_predictions.json');assert not frozen['annotations_read']
    config=read(ROOT/'baseline/e024_benchmark.json');expected={(n,t) for n in config['videos'] for t in config['frames']}
    assert {(r['video'],r['frame']) for r in inference['frames']}==expected
    assert {(r['video'],r['frame']) for r in evaluation['frames']}==expected
    assert {(r['frame']['video'],r['frame']['frame']) for r in frozen['frames']}==expected
    for k,v in evaluation['totals'].items():assert sum(r[k] for r in evaluation['frames'])==v
    totals=evaluation['totals'];assert totals['harmonic_matched']<=totals['gt_nodes'] and totals['nucverse_matched']<=totals['gt_nodes']
    result=dict(status='complete',gpu_source_files_verified=a,cpu_source_files_verified=b,totals=totals,
        gpu_seconds=inference['seconds'],cpu_seconds=evaluation['seconds'],frames=evaluation['frames'],
        extra_union_matches=totals['union_coverage_upper_bound']-totals['harmonic_matched'],
        node_count_ratio=totals['nucverse_nodes']/max(1,totals['harmonic_nodes']),
        scope=evaluation['scope'],result_sha256=sha(cpu/'nucverse_evaluation/result.json'),leaderboard_submitted=False)
    (ROOT/'results/E024_completed.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
