"""Weak 3D supervision: annotated positive cores, dark pixels, unknown elsewhere."""
import numpy as np
from scipy.spatial import cKDTree
SCALE=np.array([1.625,.40625,.40625])

def supervision(image,points,positive_um=1.2,flow_um=2.5):
    image=np.asarray(image,np.float32);points=np.asarray(points,float).reshape(-1,3)
    if image.ndim!=3 or not len(points) or not np.isfinite(points).all():raise ValueError('Invalid weak-label crop')
    grid=np.indices(image.shape,dtype=np.float32).reshape(3,-1).T
    distance,index=cKDTree(points*SCALE).query(grid*SCALE)
    positive=(distance<=positive_um).reshape(image.shape)
    flow_mask=(distance<=flow_um).reshape(image.shape)
    # Photometric weak background is explicitly identified, never all unlabeled voxels.
    dark=(image<=np.percentile(image,5)) & (image<.05)
    negative=dark & ~flow_mask
    target=(.3*(points[index]-grid)).reshape(image.shape+(3,)).astype(np.float32)
    target[~flow_mask]=0
    if not positive.any():raise ValueError('No annotated positive core in crop')
    return dict(positive=positive,negative=negative,flow_mask=flow_mask,flow=target.astype(np.float16))
