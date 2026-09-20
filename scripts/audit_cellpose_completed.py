"""Audit a paired two-frame Cellpose-DINO pilot without claiming a tracking gain."""
import json
from audit_e017_completed import ROOT,read,sha,verify
def main():
    gpu=ROOT/'outputs/e028_gpu_recovery';cpu=ROOT/'outputs/e028_eval_recovery'
    n=verify(gpu,'kaggle/cellpose_inference','results/E028_GPU_launch.json','cellpose_inference_package');m=verify(cpu,'kaggle/cellpose_evaluation','results/E028_EVAL_launch.json','cellpose_evaluation_package')
    fields=read(gpu/'cellpose_fields/result.json');result=read(cpu/'cellpose_evaluation/result.json');config=read(ROOT/'baseline/e028_pilot.json')
    assert fields['status']==result['status']=='complete' and not fields['annotations_read'] and fields['optimizer_steps']==0
    assert fields['config']==config and fields['weight_sha256']==read(ROOT/'results/E028_PREP_completed.json')['weight_sha256']
    expected={(r['video'],r['frame']) for r in config['frames']};assert len(expected)==2
    assert {(r['video'],r['frame']) for r in fields['frames']}==expected=={(r['video'],r['frame']) for r in result['frames']}
    assert not read(cpu/'cellpose_evaluation/frozen_predictions.json')['annotations_read']
    for key,total in result['totals'].items():assert sum(row[key] for row in result['frames'])==total
    old=read(ROOT/'results/E027_completed.json');lookup={(r['video'],r['frame']):r for r in old['frames_detail']};comparisons=[]
    for row in result['frames']:
        previous=lookup[(row['video'],row['frame'])]
        for key in ['gt_nodes','harmonic_nodes','harmonic_matched']:assert row[key]==previous[key]
        comparisons.append(dict(**row,nucverse_nodes=previous['nucverse_nodes'],nucverse_matched=previous['nucverse_matched'],nucverse_only=previous['nucverse_only']))
    receipt=dict(status='complete',inference_source_files_verified=n,evaluation_source_files_verified=m,frames=comparisons,totals=result['totals'],gpu_process_seconds=fields['seconds'],cpu_seconds=result['seconds'],result_sha256=sha(cpu/'cellpose_evaluation/result.json'),scope=result['scope'],leaderboard_submitted=False)
    (ROOT/'results/E028_completed.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
