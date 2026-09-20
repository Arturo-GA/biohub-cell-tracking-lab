"""Audit E027 full-frame complementary coverage; never promote by oracle union."""
import json
from audit_e017_completed import ROOT,read,sha,verify
def main():
    gpu=ROOT/'outputs/e027_gpu_recovery';cpu=ROOT/'outputs/e027_eval_recovery'
    n=verify(gpu,'kaggle/error_window_inference','results/E027_GPU_launch.json','error_window_inference_package');m=verify(cpu,'kaggle/error_window_evaluation','results/E027_EVAL_launch.json','error_window_evaluation_package')
    fields=read(gpu/'nucverse_fields/result.json');result=read(cpu/'nucverse_evaluation/result.json');inputs=read(ROOT/'outputs/e027_inputs_recovery/nucverse_inputs/result.json')
    assert fields['status']==result['status']=='complete' and fields['optimizer_steps']==0
    assert fields['moving_statistics_unchanged'] and not fields['annotations_read']
    assert fields['weight_sha256']==read(ROOT/'baseline/e026_normalization.json')['weight_sha256']
    expected={(r['video'],r['frame']):r for r in inputs['frames']}
    assert {(r['video'],r['frame']) for r in fields['frames']}==set(expected)
    for row in fields['frames']:assert all(row[k]==expected[(row['video'],row['frame'])][k] for k in ['file','shape','sha256'])
    assert {(r['video'],r['frame']) for r in result['frames']}==set(expected)
    assert not read(cpu/'nucverse_evaluation/frozen_predictions.json')['annotations_read']
    for key,total in result['totals'].items():assert sum(r[key] for r in result['frames'])==total
    totals=result['totals'];missing=totals['gt_nodes']-totals['harmonic_matched']
    assert 0<=totals['nucverse_only']<=missing
    receipt=dict(status='complete',inference_source_files_verified=n,evaluation_source_files_verified=m,frames=len(expected),totals=totals,control_missed_centers=missing,candidate_recovers_control_misses=totals['nucverse_only'],recovery_fraction=totals['nucverse_only']/missing if missing else None,gpu_process_seconds=fields['seconds'],cpu_seconds=result['seconds'],frames_detail=result['frames'],result_sha256=sha(cpu/'nucverse_evaluation/result.json'),scope='Error-conditioned calibration coverage, not unbiased recall, tracking improvement or leaderboard score; union is an oracle diagnostic',leaderboard_submitted=False)
    (ROOT/'results/E027_completed.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='frames_detail'},indent=2))
if __name__=='__main__':main()
