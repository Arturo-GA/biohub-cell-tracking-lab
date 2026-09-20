"""Verify public Cellpose preparation before any competition inference."""
import json
from audit_e017_completed import ROOT,read,sha,verify
def main():
    folder=ROOT/'outputs/e028_prep_recovery';count=verify(folder,'kaggle/cellpose_prepare','results/E028_PREP_launch.json','cellpose_prepare_package')
    path=folder/'cellpose_assets/result.json';report=read(path)
    assert report['status']=='complete' and report['strict_load'] and not report['gpu'] and not report['competition_data_read']
    assert report['config']==read(ROOT/'baseline/e028_cellpose.json')
    assert report['output_shape']==[1,3,64,64] and len(report['wheels'])>=8
    assert len(report['weight_sha256'])==64 and all(len(r['sha256'])==64 for r in report['wheels'])
    receipt=dict(report,source_files_verified=count,result_sha256=sha(path))
    (ROOT/'results/E028_PREP_completed.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
