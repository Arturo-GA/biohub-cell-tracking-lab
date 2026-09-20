"""Memory-bounded NucVerse adapter with explicitly uniform overlap blending."""
import itertools,time
import numpy as np

def predict_volume(predict,image,patch=(64,128,128),stride=(48,96,96),deadline=None):
    image=np.asarray(image,dtype=np.float32)
    if image.ndim!=3 or not np.isfinite(image).all():raise ValueError('Invalid volume')
    if any(s<=0 or s>p for p,s in zip(patch,stride)):raise ValueError('Invalid tiling stride')
    low,high=np.percentile(image,[2,99.8])
    image=np.clip((image-low)/max(float(high-low),1e-8),0,1)
    shape=image.shape
    extent=tuple(p+max(0,int(np.ceil((n-p)/s)))*s for n,p,s in zip(shape,patch,stride))
    padded=np.pad(image,[(0,n-m) for n,m in zip(extent,shape)],mode='reflect')
    result=np.zeros(extent+(4,),np.float32);count=np.zeros(extent,np.uint16);tiles=0
    for start in itertools.product(*[range(0,n-p+1,s) for n,p,s in zip(extent,patch,stride)]):
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Frozen GPU runtime budget exceeded')
        slices=tuple(slice(a,a+p) for a,p in zip(start,patch))
        binary,gradient=[np.asarray(v) for v in predict(padded[slices][None,...,None])]
        assert binary.shape==(1,)+tuple(patch)+(2,) and gradient.shape==(1,)+tuple(patch)+(3,)
        assert np.isfinite(binary).all() and np.isfinite(gradient).all()
        result[slices]+=np.concatenate([binary[0,...,1:2],gradient[0]],axis=-1);count[slices]+=1;tiles+=1
    assert np.all(count>0)
    result/=count[...,None]
    crop=tuple(slice(0,n) for n in shape)
    return result[crop],dict(tiles=tiles,normalization_percentiles=[2,99.8],blend='uniform overlap average',shape=list(shape))
