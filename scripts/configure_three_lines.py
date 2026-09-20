"""Freeze cohorts before model outcomes; preserve original calibration videos."""
import hashlib,json
from pathlib import Path
from build_three_lines import build
ROOT=Path(__file__).resolve().parents[1]
def main():
    audit=json.loads((ROOT/'results/E038_AUDIT_completed.json').read_text())['result'];old=json.loads((ROOT/'baseline/e035_division_sequence.json').read_text());dev=[]
    for embryo in ['44b6','6bba']:
        pool=[r for r in audit['videos'] if r['video'].startswith(embryo) and not r['previous_cohort']]
        order=lambda r:hashlib.sha256(('three-lines-v1/'+r['video']).encode()).hexdigest()
        positive=sorted([r for r in pool if r['divisions']],key=order)[:6]
        other=sorted([r for r in pool if not r['divisions']],key=order)[:8-len(positive)]
        dev.extend(r['video'] for r in positive+other)
    excluded=set(dev+old['validation']+old['development']);expanded=sorted(r['video'] for r in audit['videos'] if r['video'] not in excluded)
    cfg=dict(experiment='E038',seed=380920,fit=old['fit'],development=sorted(dev),expanded_fit=expanded,old_calibration=old['validation'],weight_sha256=audit['spatialdino']['sha256'],source_commit='ca3ab86b34430d963f12a3909baaeb9343c63b7d',protocol='Frozen SpatialDINO native3D localization probe: max128 random annotated centers/video, fixed uniform +/-4 isotropic voxel offset. 16cube interpolated3x; spatial2cube pooled tokens3072D, raw8cube512D. Ridge alpha100 fit40 videos. New development16 outside E030-E037 cohort, enriched for divisions using annotation counts only. Pass requires >10% lower mean per-video localization error vs raw ridge and zero displacement and improvement in both embryos. Shared images for subsequent E039 and E040; no leaderboard submission. Upstream pretrained membership unknown; these are development videos, not a final holdout.')
    assert not set(cfg['fit'])&set(dev)
    p=ROOT/'baseline/e038_protocol.json';assert not p.exists();p.write_text(json.dumps(cfg,indent=2)+'\n')
    files=['baseline/e038_protocol.json','src/biohub_lab/temporal_data.py']
    build('E038_PREPARE','scripts/three_lines_prepare_runner.py',files,[])
    build('E038_FEATURES','scripts/spatial_probe_features_runner.py',['baseline/e038_protocol.json','src/biohub_lab/spatial_probe.py'],['jarturo/biohub-e038-prepare-cpu','jarturo/biohub-e038-audit-cpu'],gpu=True)
    build('E038_EVALUATE','scripts/spatial_probe_evaluate_runner.py',['baseline/e038_protocol.json'],['jarturo/biohub-e038-features'])
    print('Development divisions',sum(r['divisions'] for r in audit['videos'] if r['video'] in dev),'expanded fit divisions',sum(r['divisions'] for r in audit['videos'] if r['video'] in expanded),'expanded videos',len(expanded))
if __name__=='__main__':main()
