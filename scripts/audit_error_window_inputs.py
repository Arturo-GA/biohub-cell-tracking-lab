"""Audit error-conditioned window membership and payload before GPU inference."""
import json
from audit_e017_completed import ROOT,read,sha,verify
def main():
    folder=ROOT/'outputs/e027_inputs_recovery';n=verify(folder,'kaggle/error_window_inputs','results/E027_INPUTS_launch.json','error_window_inputs_package')
    path=folder/'nucverse_inputs/result.json';report=read(path);config=read(ROOT/'baseline/e027_error_windows.json')
    assert report['status']=='complete' and report['selection']==config and report['annotations_used_to_select_frames']
    assert len(report['windows'])==len(config['videos'])==16
    assert {r['video'] for r in report['windows']}==set(config['videos'])
    expected=set();missing=0;division_missing=0
    for window in report['windows']:
        if 'skipped' in window:continue
        assert window['anchor']['missing']>0
        assert window['anchor']['frame'] in window['frames'] and len(window['frames'])<=3
        expected.update((window['video'],t) for t in window['frames'])
        missing+=window['anchor']['missing'];division_missing+=window['anchor']['missing_division']
    assert {(r['video'],r['frame']) for r in report['frames']}==expected and len(report['frames'])==len(expected)<=48
    receipt=dict(status='complete',source_files_verified=n,manifest_sha256=sha(path),frames=len(expected),videos=len({r[0] for r in expected}),anchor_missing_centers=missing,anchor_missing_division_relatives=division_missing,seconds=report['seconds'],windows=report['windows'],scope=report['scope'])
    (ROOT/'results/E027_INPUTS_completed.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
