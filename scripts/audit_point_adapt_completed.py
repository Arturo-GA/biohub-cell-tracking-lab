"""Audit E025 training provenance and the paired full-frame detection assessment."""
import json,math
from audit_e017_completed import ROOT,read,sha,verify
def main():
    train=ROOT/'outputs/e025_train_recovery';cpu=ROOT/'outputs/e025_eval_recovery'
    n=verify(train,'kaggle/point_adapt_train','results/E025_TRAIN_launch.json','point_adapt_train_package')
    m=verify(cpu,'kaggle/point_adapt_evaluation','results/E025_EVAL_launch.json','point_adapt_evaluation_package')
    training=read(train/'point_adaptation/result.json');fields=read(train/'nucverse_fields/result.json');result=read(cpu/'nucverse_evaluation/result.json')
    assert training['status']==fields['status']==result['status']=='complete'
    config=read(ROOT/'baseline/e025_point_adapt.json');data=read(ROOT/'results/E025_DATA_completed.json')
    assert training['training_manifest_sha256']==data['manifest_sha256']
    assert training['training_videos']==config['fit'] and training['validation_videos']==config['validation']
    assert not set(config['fit'])&set(config['validation']) and training['steps']==config['steps']
    history=read(train/'point_adaptation/history.json');assert history==training['history']
    assert [r['step'] for r in history]==[0,200,400,600,800]
    assert all(math.isfinite(v) for r in history for v in r['validation'])
    assert fields['weight_sha256']==training['weight_sha256'] and not fields['annotations_read']
    frozen=read(cpu/'nucverse_evaluation/frozen_predictions.json');assert not frozen['annotations_read']
    before=read(ROOT/'results/E024_completed.json');lookup={(r['video'],r['frame']):r for r in before['frames']}
    assert {(r['video'],r['frame']) for r in result['frames']}==set(lookup)
    for r in result['frames']:
        previous=lookup[(r['video'],r['frame'])]
        for k in ['gt_nodes','harmonic_nodes','harmonic_matched']:assert r[k]==previous[k]
    for k,v in result['totals'].items():assert sum(r[k] for r in result['frames'])==v
    receipt=dict(status='complete',training_source_files_verified=n,evaluation_source_files_verified=m,
        training_seconds=training['training_seconds'],total_gpu_process_seconds=training['total_seconds'],cpu_seconds=result['seconds'],
        history=history,weight_sha256=training['weight_sha256'],original_totals=before['totals'],adapted_totals=result['totals'],
        result_sha256=sha(cpu/'nucverse_evaluation/result.json'),scope=result['scope'],leaderboard_submitted=False)
    (ROOT/'results/E025_completed.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
