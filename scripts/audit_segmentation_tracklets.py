"""Verify the actual E029 CSV, fixed graph proposal and official metric aggregates."""
import csv,json,math
from collections import Counter
from audit_e017_completed import ROOT,read,sha,verify
from audit_cpu_control import aggregate
def main():
    folder=ROOT/'outputs/e029_recovery';count=verify(folder,'kaggle/segmentation_tracklets','results/E029_launch.json','segmentation_tracklets_package');root=folder/'segmentation_tracklets'
    report=read(root/'result.json');metric=read(root/'candidate_metrics.json');frozen=read(root/'frozen_predictions.json');recovery=read(root/'recovery_reports.json')
    assert report['status']=='complete' and frozen['donor_frames_annotation_selected'] and not frozen['calibration_annotations_read']
    assert sha(root/'candidate.csv')==frozen['csv_sha256']
    before=read(ROOT/'outputs/e017_compare_recovery/calibration_compare/harmonic_metrics.json');old={r['dataset']:r for r in before['samples']}
    assert set(recovery)==set(old)==set(frozen['videos'])
    for key,value in aggregate(metric['samples']).items():assert math.isclose(value,metric['summary'][key],rel_tol=0,abs_tol=1e-12)
    counts=Counter()
    with (root/'candidate.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['row_type']=='node':counts[row['dataset']]+=1
    for row in metric['samples']:assert row['num_pred_nodes']==counts[row['dataset']]==old[row['dataset']]['num_pred_nodes']+recovery[row['dataset']]['added_nodes']
    for key in ['added_nodes','added_edges']:assert report[key]==sum(r[key] for r in recovery.values())
    assert math.isclose(report['score_delta'],metric['summary']['score']-before['summary']['score'],abs_tol=1e-12)
    receipt=dict(status='complete',source_files_verified=count,csv_sha256=frozen['csv_sha256'],candidate=metric['summary'],control=before['summary'],score_delta=report['score_delta'],added_nodes=report['added_nodes'],added_edges=report['added_edges'],seconds=report['seconds'],scope=report['scope'],leaderboard_submitted=False,independent_validation=False)
    (ROOT/'results/E029_completed.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
