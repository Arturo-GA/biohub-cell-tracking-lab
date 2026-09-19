"""CPU diagnostics and annotation-free projection of competing detections."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

SCALE=np.array([1.625,.40625,.40625])


def mappings(coords,truth,radius=7.,ambiguity=1.):
    """Training-like many-to-one vs max-cardinality minimum-distance diagnostic.

    The latter is a diagnostic assignment, not a reimplementation of tracksdata.
    No unmatched prediction is classified as a background/negative cell.
    """
    many=np.full(len(coords),-1,np.int64);one=many.copy()
    for frame in np.unique(coords[:,0]):
        ids=np.flatnonzero(coords[:,0]==frame);refs=np.flatnonzero(truth[:,0]==frame)
        if not len(refs):continue
        distances,local=cKDTree(truth[refs,1:]*SCALE).query(coords[ids,1:]*SCALE,k=2)
        good=(distances[:,0]<=radius)&((distances[:,1]-distances[:,0])>=ambiguity)
        many[ids[good]]=refs[local[good,0]]
        costs=cdist(truth[refs,1:]*SCALE,coords[ids,1:]*SCALE)
        dummy=(len(refs)+1)*(radius+1)
        costs=np.where(costs<=radius,costs,2*dummy)
        rows,cols=linear_sum_assignment(np.c_[costs,np.full((len(refs),len(refs)),dummy)])
        valid=cols<len(ids);rows,cols=rows[valid],cols[valid]
        valid=costs[rows,cols]<=radius
        one[ids[cols[valid]]]=refs[rows[valid]]
    return many,one


def identity_audit(coords,edges,truth,gt_edges):
    many,one=mappings(coords,truth)
    counts=np.bincount(many[many>=0],minlength=len(truth))
    known=set(map(tuple,gt_edges.tolist()))
    def count(mapping):
        pairs=mapping[edges];valid=(pairs>=0).all(1)
        positives=[tuple(e) for e in pairs[valid] if tuple(e) in known]
        return dict(mapped_edge_instances=int(valid.sum()),positive_edge_instances=len(positives),
                    distinct_positive_gt_edges=len(set(positives)))
    return dict(predicted_nodes=len(coords),annotated_nodes=len(truth),
        many_to_one_matched_predictions=int((many>=0).sum()),
        many_to_one_covered_gt=int((counts>0).sum()),gt_with_multiple_predictions=int((counts>1).sum()),
        largest_multiplicity=int(counts.max(initial=0)),one_to_one_matched_predictions=int((one>=0).sum()),
        many_to_one_edges=count(many),one_to_one_edges=count(one),
        scope='Diagnostic correspondences only; not an official score or labels for prediction')


def maximum_vote_edges(coords,edges,votes):
    chosen=[]
    for frame in np.unique(coords[edges[:,0],0]) if len(edges) else []:
        ids=np.flatnonzero(coords[edges[:,0],0]==frame);e=edges[ids]
        mothers,mi=np.unique(e[:,0],return_inverse=True);daughters,di=np.unique(e[:,1],return_inverse=True)
        distance=np.linalg.norm((coords[e[:,1],1:]-coords[e[:,0],1:])*SCALE,axis=1)
        utility=votes[ids].astype(float)+.01/(1+distance)
        offset=float(utility.max()+1)
        matrix=coo_matrix((np.r_[offset-utility,np.full(len(mothers),offset)],
            (np.r_[mi,np.arange(len(mothers))],np.r_[di,len(daughters)+np.arange(len(mothers))])),
            shape=(len(mothers),len(daughters)+len(mothers))).tocsr()
        row,col=min_weight_full_bipartite_matching(matrix)
        valid=col<len(daughters)
        chosen.extend(zip(mothers[row[valid]],daughters[col[valid]]))
    return np.array(sorted(chosen),np.int64).reshape(-1,2)


def project_detections(coords,edges,origin,radius=3.):
    """Select physically exclusive representatives and globally match edge votes.

    Fixed Harmonic-origin priority, then temporal component span, then original ID.
    No annotation, model retraining, node-count metadata or threshold search.
    """
    active=np.unique(edges);parent=np.arange(len(coords));span=np.zeros(len(coords))
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for a,b in edges:
        x,y=find(int(a)),find(int(b))
        if x!=y:parent[y]=x
    groups={}
    for node in active:groups.setdefault(find(int(node)),[]).append(node)
    for ids in groups.values():span[ids]=np.ptp(coords[ids,0])+1
    representative=np.full(len(coords),-1,np.int64)
    for frame in np.unique(coords[active,0]):
        ids=active[coords[active,0]==frame];tree=cKDTree(coords[ids,1:]*SCALE)
        order=np.lexsort((ids,-span[ids],origin[ids]!=0))
        for index in order:
            node=ids[index]
            if representative[node]>=0:continue
            nearby=ids[tree.query_ball_point(coords[node,1:]*SCALE,radius)]
            free=nearby[representative[nearby]<0];representative[free]=node
    collapsed=representative[edges]
    unique,votes=np.unique(collapsed,axis=0,return_counts=True)
    selected=maximum_vote_edges(coords,unique,votes)
    if len(selected):
        assert np.all(coords[selected[:,1],0]==coords[selected[:,0],0]+1)
        assert np.bincount(selected[:,0]).max()<=1 and np.bincount(selected[:,1]).max()<=1
    return selected,dict(input_nodes=len(active),representatives=len(np.unique(representative[active])),
        output_nodes=len(np.unique(selected)),input_edges=len(edges),output_edges=len(selected),
        radius_um=radius,annotations_used=False)
