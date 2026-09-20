"""Verify the paired CPU normalization diagnostic, without claiming tracking gains."""
import json,math
from audit_e017_completed import ROOT,read,sha,verify

def main():
    folder=ROOT/'outputs/e025_norm_recovery'
    count=verify(folder,'kaggle/point_norm','results/E025_NORM_launch.json','point_norm_package')
    path=folder/'normalization_diagnostic/result.json';report=read(path)
    assert report['status']=='complete' and not report['gpu'] and report['optimizer_steps']==0
    assert report['state_restored_before_each_forward'] and report['dropout_disabled']>0
    assert len(report['rows'])==32 and len(report['groups'])==8
    config=read(ROOT/'baseline/e025_point_adapt.json')
    groups=[]
    for row in report['groups']:
        rows=[r for r in report['rows'] if all(r[k]==row[k] for k in ['model','split','mode'])]
        assert len(rows)==row['n']==4
        assert len({r['video'] for r in rows})==4
        assert all(r['video'] in config['fit' if row['split']=='fit' else 'validation'] for r in rows)
        for key in ['positive_bce','positive_probability']:
            assert all(math.isfinite(r[key]) for r in rows)
            assert abs(sum(r[key] for r in rows)/len(rows)-row[key])<1e-8
        groups.append(row)
    result=dict(status='complete',source_files_verified=count,result_sha256=sha(path),groups=groups,seconds=report['seconds'],scope=report['scope'],leaderboard_submitted=False)
    (ROOT/'results/E025_NORM_completed.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
