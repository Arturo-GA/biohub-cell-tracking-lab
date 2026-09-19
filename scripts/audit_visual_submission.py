"""Audit the completed inference package and locally validate its final CSV."""
import json
from audit_e017_completed import ROOT,read,sha,verify
from biohub_lab.submission import read_and_validate

def main():
    folder=ROOT/'outputs/visual_submission_recovery'
    assert read(folder/'status.json')['status']=='COMPLETE'
    count=verify(folder,'kaggle/visual_submission','results/VISUAL_SUBMISSION_launch.json','visual_submission_package')
    result=read(folder/'visual_submission/result.json')
    post=read(folder/'visual_submission/postprocess.json')
    assert result['status']==post['status']=='complete' and result['submission_ready']
    assert not result['annotations_read'] and not post['annotations_read']
    csv=folder/'submission.csv'
    assert sha(csv)==result['csv_sha256']==post['csv_sha256']
    groups=read_and_validate(csv,post['image_shapes'])
    assert set(groups)==set(post['videos'])
    for name,(nodes,edges) in groups.items():
        assert len(nodes)==post['videos'][name]['nodes'] and len(edges)==post['videos'][name]['edges']
    receipt=dict(status='complete',source_files_verified=count,csv_sha256=sha(csv),
        csv_locally_validated=True,videos=len(groups),seconds=result['seconds'],
        launch_receipt_sha256=sha(ROOT/'results/VISUAL_SUBMISSION_launch.json'),leaderboard_submitted=False)
    (ROOT/'results/VISUAL_SUBMISSION_completed.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
