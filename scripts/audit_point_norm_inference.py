"""Audit E026 unchanged weights and paired full-frame normalization experiment."""
import json
from audit_e017_completed import ROOT,read,sha,verify
def main():
    gpu=ROOT/'outputs/e026_gpu_recovery';cpu=ROOT/'outputs/e026_eval_recovery'
    n=verify(gpu,'kaggle/point_norm_inference','results/E026_GPU_launch.json','point_norm_inference_package')
    m=verify(cpu,'kaggle/point_norm_evaluation','results/E026_EVAL_launch.json','point_norm_evaluation_package')
    fields=read(gpu/'nucverse_fields/result.json');result=read(cpu/'nucverse_evaluation/result.json')
    assert fields['status']==result['status']=='complete'
    assert not fields['annotations_read'] and fields['optimizer_steps']==0 and fields['moving_statistics_unchanged']
    previous=read(ROOT/'results/E025_completed.json')
    assert fields['weight_sha256']==previous['weight_sha256']
    assert not read(cpu/'nucverse_evaluation/frozen_predictions.json')['annotations_read']
    before=read(ROOT/'results/E024_completed.json');lookup={(r['video'],r['frame']):r for r in before['frames']}
    assert {(r['video'],r['frame']) for r in result['frames']}==set(lookup)
    for row in result['frames']:
        old=lookup[(row['video'],row['frame'])]
        for key in ['gt_nodes','harmonic_nodes','harmonic_matched']:assert row[key]==old[key]
    for key,total in result['totals'].items():assert sum(r[key] for r in result['frames'])==total
    receipt=dict(status='complete',inference_source_files_verified=n,evaluation_source_files_verified=m,weight_sha256=fields['weight_sha256'],optimizer_steps=0,moving_statistics_unchanged=True,gpu_process_seconds=fields['seconds'],cpu_seconds=result['seconds'],original_totals=before['totals'],adapted_stored_statistics_totals=previous['adapted_totals'],adapted_batch_statistics_totals=result['totals'],result_sha256=sha(cpu/'nucverse_evaluation/result.json'),scope=result['scope'],leaderboard_submitted=False)
    (ROOT/'results/E026_completed.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
