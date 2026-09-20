"""Verify executed E030 source, split isolation and chained stage hashes."""
import json,sys
from audit_e017_completed import ROOT,read,sha,verify


def main(stage):
    folder=ROOT/('outputs/e030_'+stage);path=folder/('image_identity_'+stage+'/result.json')
    result=read(path);assert result['status']=='complete' and read(folder/'status.json')['status']=='COMPLETE'
    n=verify(folder,'kaggle/image_identity_'+stage,'results/E030_'+stage.upper()+'_launch.json','image_identity_'+stage+'_package')
    config=read(ROOT/'baseline/e030_image_identity.json');assert result['config']==config
    assert not set(config['fit'])&set(config['validation'])
    if stage in ('data','features'):
        for split in ('fit','validation'):assert {r['video'] for r in result['videos'] if r['split']==split}==set(config[split])
    if stage=='features':
        assert result['optimizer_steps']==0 and not result['styles_used'] and result['deterministic_max_error']<1e-6
        assert result['input_manifest_sha256']==read(ROOT/'results/E030_DATA_completed.json')['result_sha256']
        assert result['weight_sha256']==read(ROOT/'results/E028_PREP_completed.json')['weight_sha256']
    if stage=='train':
        assert result['input_manifest_sha256']==read(ROOT/'results/E030_DATA_completed.json')['result_sha256']
        assert result['feature_manifest_sha256']==read(ROOT/'results/E030_FEATURES_completed.json')['result_sha256']
        for name in ('image','geometry_ablation'):
            value=result['results'][name];assert value['training']['steps']==config['steps']
            r=value['validation'];assert {v['video'] for v in r['videos']}==set(config['validation'])
            assert r['targets']==sum(v['targets'] for v in r['videos'])
            assert abs(r['parent_accuracy']-sum(v['correct'] for v in r['videos'])/max(r['targets'],1))<1e-12
    receipt=dict(status='complete',stage=stage,source_files_verified=n,result_sha256=sha(path),seconds=result['seconds'],leaderboard_submitted=False,scope=result.get('scope','Frozen features from annotation-selected crops; no accuracy claim'))
    if stage=='data':receipt.update(nodes=sum(v['nodes'] for v in result['videos']),fit_division_positive=sum(v['division_positive'] for v in result['videos'] if v['split']=='fit'),validation_division_positive=sum(v['division_positive'] for v in result['videos'] if v['split']=='validation'))
    if stage=='train':receipt.update(results=result['results'],image_parent_delta_vs_geometry=result['image_parent_delta_vs_geometry'])
    (ROOT/('results/E030_'+stage.upper()+'_completed.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main(sys.argv[1])
