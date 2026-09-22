"""CPU-only visible-test change check; cached control for research, never submitted."""
import json,sys,time
from pathlib import Path
import zarr
from residual_detector_io import one,sha
from biohub_lab.appearance_swap import describe,refine
from biohub_lab.submission import read_and_validate

def main(package):
    start=time.monotonic();cfg=json.loads((package/'baseline/e065_preview.json').read_text());p=one('native_centers_submission/control.csv');assert sha(p)==cfg['control_sha256']
    base=json.loads((p.parent/'result.json').read_text());graphs=read_and_validate(p,base['shapes'])
    test=next(p/'test' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'test').exists());reports={a:dict(changed_edges=0,videos=[]) for a in cfg['arms']}
    for v,(nodes,edges) in graphs.items():
        features=describe(nodes,zarr.open_group(str(test/(v+'.zarr')),mode='r')['0'])
        for a,options in cfg['arms'].items():
            es,r=refine(nodes,edges,features,**options);reports[a]['changed_edges']+=r['changed_edges'];reports[a]['videos'].append(dict(video=v,**r))
        print('ENSEMBLE_APPEARANCE_PREVIEW',v,{a:reports[a]['changed_edges'] for a in reports},flush=True)
    out=Path('/kaggle/working/appearance_swap_preview');out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(dict(status='complete',reports=reports,seconds=time.monotonic()-start,scope='Visible-test label-free change audit only. No submission CSV exported.'),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
