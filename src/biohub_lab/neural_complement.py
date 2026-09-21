"""Extract persistent chains from neural global frame assignments."""
import numpy as np
from biohub_lab.detection_identity import maximum_vote_edges
from biohub_lab.residual_tracklets import SCALE

def neural_chains(coords,edges,prob):
    assert len(edges)==len(prob) and np.isfinite(prob).all()
    distance=np.linalg.norm((coords[edges[:,1],1:]-coords[edges[:,0],1:])*SCALE,axis=1)
    keep=(distance<=14)&(prob>.05)
    chosen=maximum_vote_edges(coords,edges[keep],np.log(prob[keep]/.05))
    forward={int(a):int(b) for a,b in chosen};incoming={int(b) for a,b in chosen};paths=[]
    for start in sorted(set(forward)-incoming):
        path=[start]
        while path[-1] in forward:
            child=forward[path[-1]];assert coords[child,0]==coords[path[-1],0]+1;path.append(child)
        if len(path)>=5:paths.append(path)
    return paths
