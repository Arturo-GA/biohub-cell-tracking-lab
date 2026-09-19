"""Set accounting for recovered annotated links; never a prediction fusion rule."""
import numpy as np

def overlap(reference,candidate,truth,reference_nodes):
    reference,candidate,truth=set(reference),set(candidate),set(truth)
    if not reference<=truth or not candidate<=truth:raise ValueError('TP set outside ground truth')
    common=reference&candidate;rescued=candidate-reference;lost=reference-candidate
    missing_detection=sum(a not in reference_nodes or b not in reference_nodes for a,b in rescued)
    return dict(gt_edges=len(truth),both_correct=len(common),harmonic_only=len(lost),tissue_only=len(rescued),
        neither_correct=len(truth-(reference|candidate)),harmonic_tp=len(reference),tissue_tp=len(candidate),
        rescued_with_both_endpoints_matched_in_harmonic=len(rescued)-missing_detection,
        rescued_with_missing_harmonic_endpoint=missing_detection,
        diagnostic_union_tp=len(reference|candidate),
        scope='Ground-truth set overlap only; union is not a realizable fused graph or an official score')

def feature_bins(nodes,edges):
    """Test-time-available geometry; bins fixed before looking at recovery labels."""
    scale=np.array([1.625,.40625,.40625]);parents={};children={}
    for a,b in edges:parents.setdefault(b,[]).append(a);children.setdefault(a,[]).append(b)
    result={}
    for a,b in edges:
        x=np.array([nodes[a][k] for k in ('z','y','x')])*scale
        y=np.array([nodes[b][k] for k in ('z','y','x')])*scale
        displacement=y-x;distance=np.linalg.norm(displacement)
        label='distance_le3' if distance<=3 else 'distance_3to6' if distance<=6 else 'distance_gt6'
        history=parents.get(a,[]);future=children.get(b,[])
        context='two_sided_context' if len(history)==len(future)==1 else 'incomplete_or_branched_context'
        acceleration='acceleration_unknown'
        if len(history)==1:
            p=np.array([nodes[history[0]][k] for k in ('z','y','x')])*scale
            acceleration='acceleration_le2' if np.linalg.norm(displacement-(x-p))<=2 else 'acceleration_gt2'
        result[(a,b)]=(label,context,acceleration)
    return result
