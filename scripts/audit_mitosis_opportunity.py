"""Describe exact-frame geometric repair opportunities; this is not an official oracle."""
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from biohub_lab.temporal_data import SCALE


def audit(root=Path('outputs/mitosis_train'),truth_root=Path('outputs/mitosis_audit/temporal_cache')):
    root=Path(root);truth_root=Path(truth_root)
    receipt=json.loads((root/'mitosis_experiment_receipt.json').read_text());rows=[];hashes={}
    for name in receipt['diagnostic']['statistics']:
        truth_path=truth_root/name/'graph.npz'
        hashes[name]=hashlib.sha256(truth_path.read_bytes()).hexdigest()
        with np.load(truth_path) as archive: truth={k:archive[k] for k in archive.files}
        cache_receipt=json.loads((truth_path.parent/'receipt.json').read_text())
        config_hash=hashlib.sha256(json.dumps(cache_receipt['config'],sort_keys=True).encode()+
            truth['coords'][:cache_receipt['gt_nodes']].tobytes()+truth['edges'].tobytes()).hexdigest()
        if config_hash!=cache_receipt['config_hash']: raise ValueError('Prepared GT graph/config checksum mismatch')
        with np.load(root/'mitosis_diagnostic'/(name+'_scores.npz')) as archive: pred={k:archive[k] for k in archive.files}
        coords=pred['coords'];tri=pred['triples'];baseline=pred['baseline']
        children=defaultdict(list);parents={}
        for m,d in baseline:children[int(m)].append(int(d));parents[int(d)]=int(m)
        group=name.split('_')[0];threshold=receipt['groups'][group]['threshold']
        with np.load(root/'mitosis_models'/group/'holdout_scores.npz') as archive:
            holdout={k:archive[k] for k in archive.files}
        for ref in np.flatnonzero(truth['triple_labels']==1):
            gt=truth['triples'][ref];points=truth['coords'][gt];near=[];near_dist=[]
            for point in points:
                indices=np.flatnonzero(coords[:,0]==point[0])
                distances=np.linalg.norm((coords[indices,1:]-point[1:])*SCALE,axis=1)
                closest=int(np.argmin(distances));distance=float(distances[closest])
                near.append(int(indices[closest]) if distance<=7. else None);near_dist.append(distance)
            mother_t=points[0,0]
            refs=np.flatnonzero(coords[tri[:,0],0]==mother_t)
            tt=tri[refs]
            dm=np.linalg.norm((coords[tt[:,0],1:]-points[0,1:])*SCALE,axis=1)
            da=np.linalg.norm((coords[tt[:,1],1:]-points[1,1:])*SCALE,axis=1)
            db=np.linalg.norm((coords[tt[:,2],1:]-points[2,1:])*SCALE,axis=1)
            da_swap=np.linalg.norm((coords[tt[:,2],1:]-points[1,1:])*SCALE,axis=1)
            db_swap=np.linalg.norm((coords[tt[:,1],1:]-points[2,1:])*SCALE,axis=1)
            geometric=refs[(dm<=7.)&(np.minimum(np.maximum(da,db),np.maximum(da_swap,db_swap))<=7.)]
            selected=set(map(int,pred['selected_repairs']));exact=[]
            if None in near:
                classification='one_or_more_nearest_detections_outside_7um'
            elif near[1]==near[2]:
                classification='both_daughters_share_nearest_detection'
            else:
                m,a,b=near
                exact=[int(i) for i in refs if tri[i,0]==m and set(tri[i,1:])=={a,b}]
                if set(children[m])=={a,b}:classification='nearest_event_already_present_in_baseline'
                elif exact:
                    if any(i in selected for i in exact):classification='nearest_event_repair_selected'
                    elif any(pred['scores'][i]>=threshold for i in exact):classification='nearest_event_lost_assignment_conflict'
                    else:classification='nearest_event_below_fixed_dev_threshold'
                elif len(children[m])!=1:classification='nearest_mother_not_a_one_child_baseline_mother'
                elif children[m][0] not in (a,b):classification='baseline_child_is_not_a_nearest_gt_daughter'
                else:
                    missing=b if children[m][0]==a else a
                    classification=('nearest_missing_daughter_already_has_a_parent' if missing in parents else
                                    'nearest_event_outside_distance_or_top4_candidate_rule')
            gt_refs=np.flatnonzero((holdout['video']==name)&(holdout['row']==ref))
            if len(gt_refs)!=1 or holdout['labels'][gt_refs[0]]!=1: raise ValueError('GT candidate score membership mismatch')
            gt_score=float(holdout['scores'][gt_refs[0]])
            row=dict(dataset=name,gt_triple_row=int(ref),gt_mother_frame=int(mother_t),gt_coords=points.tolist(),
                nearest_predicted_nodes=near,nearest_distances_um=near_dist,
                nearest_mother_baseline_children=children[near[0]] if near[0] is not None else None,
                nearest_daughter_baseline_parents=[parents.get(n) for n in near[1:]],
                classification=classification,exact_nearest_candidate_refs=exact,
                geometric_exact_frame_candidates=len(geometric),
                geometric_candidates_above_threshold=int((pred['scores'][geometric]>=threshold).sum()),
                geometric_candidates_selected=sum(int(i) in selected for i in geometric),
                maximum_geometric_candidate_score=float(pred['scores'][geometric].max()) if len(geometric) else None,
                fixed_development_threshold=threshold,score_on_GT_center_patches=gt_score,
                GT_center_score_above_threshold=gt_score>=threshold)
            rows.append(row)
    if len(rows)!=7: raise ValueError('Unexpected diagnostic division inventory')
    counts=defaultdict(int)
    for row in rows:counts[row['classification']]+=1
    result=dict(experiment='E007',checked_at_utc=datetime.now(timezone.utc).isoformat(),events=rows,
        classifications=dict(counts),gt_graph_sha256=hashes,prepared_GT_config_checksums_verified=True,
        geometric_exact_frame_eligible_events=sum(r['geometric_exact_frame_candidates']>0 for r in rows),
        GT_center_events_above_fixed_threshold=sum(r['GT_center_score_above_threshold'] for r in rows),
        scope='Post-result structural diagnostic only. Nearest detections and exact-frame 7um proximity are NOT official graph matching, which uses lineage context and temporal tolerance. Geometric eligibility is necessary only for this exact-frame interpretation and is not sufficient for an official true positive. No thresholds or predictions changed.')
    Path('results/E007_opportunity_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('events','gt_graph_sha256')}))
    for row in rows: print(row['dataset'],row['gt_mother_frame'],row['classification'],
        'geometric',row['geometric_exact_frame_candidates'],'GT_score',row['score_on_GT_center_patches'])
    return result


if __name__=='__main__': audit()
