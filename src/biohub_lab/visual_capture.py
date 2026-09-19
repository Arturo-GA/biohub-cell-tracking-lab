"""Record learned alternative links without modifying baseline inference."""
from pathlib import Path
import numpy as np

def capture_pair(name,time,source,target,source_ids,target_ids,prob,downsample,root='/kaggle/working/visual_candidate_cache'):
    prob=np.asarray(prob); assert prob.shape==(len(source_ids),len(target_ids))
    # Union of best eight mothers per daughter and best eight daughters per mother.
    keep=np.zeros(prob.shape,bool)
    if prob.size:
        k=min(8,prob.shape[1]); rows=np.arange(prob.shape[0])[:,None]
        keep[rows,np.argpartition(prob,-k,axis=1)[:,-k:]]=True
        k=min(8,prob.shape[0]);cols=np.arange(prob.shape[1])[None,:]
        keep[np.argpartition(prob,-k,axis=0)[-k:,:],cols]=True
    a,b=np.nonzero(keep)
    folder=Path(root)/name;folder.mkdir(parents=True,exist_ok=True)
    ca=np.asarray(source,dtype=np.float32).copy();cb=np.asarray(target,dtype=np.float32).copy()
    ca[:,1:]*=np.asarray(downsample);cb[:,1:]*=np.asarray(downsample)
    np.savez_compressed(folder/(str(int(time))+'.npz'),source_coords=ca,target_coords=cb,
        source_ids=source_ids,target_ids=target_ids,edges=np.c_[a,b],prob=prob[a,b].astype('float32'))

def patch_predictor(source):
    anchor='            candidates = sorted(\n'
    assert source.count(anchor)==1
    inserted='''            from biohub_lab.visual_capture import capture_pair
            capture_pair(ds_path.stem, t_src, c_src, c_tgt, idx_src, idx_tgt, probs, ds_arr)

'''
    result=source.replace(anchor,inserted+anchor)
    compile(result,'captured_predictor','exec');return result

def inject_baseline(source):
    anchor='def list_test_stems() -> list[str]:'
    assert source.count(anchor)==1
    inserted='''from biohub_lab.visual_capture import patch_predictor
_ps.write_text(patch_predictor(_ps.read_text()))

'''
    result=source.replace(anchor,inserted+anchor)
    compile(result,'capture_baseline','exec');return result
