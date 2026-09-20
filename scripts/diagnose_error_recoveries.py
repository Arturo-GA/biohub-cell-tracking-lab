"""Laptop CPU: classify error recoveries by proximity, without constructing a submission."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from biohub_lab.nucverse_instances import SCALE,matched_truth
ROOT=Path(__file__).resolve().parents[1]

def pairs(prediction,truth):
    if not len(prediction) or not len(truth):return {}
    distance=cdist(truth*SCALE,prediction*SCALE);penalty=(len(truth)+1)*8
    rows,cols=linear_sum_assignment(np.c_[np.where(distance<=7,distance,2*penalty),np.full((len(truth),len(truth)),penalty)])
    return {int(r):int(c) for r,c in zip(rows,cols) if c<len(prediction) and distance[r,c]<=7}

def main():
    root=ROOT/'outputs/e027_diagnostic_data';frozen=json.loads((ROOT/'outputs/e027_eval_recovery/nucverse_evaluation/frozen_predictions.json').read_text())
    predictions={};references={}
    for row in frozen['frames']:
        frame=row['frame'];key=frame['video'],frame['frame'];path=root/row['prediction']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        predictions[key]=np.load(path,allow_pickle=False)
        with np.load(root/(Path(frame['file']).stem+'_reference.npz'),allow_pickle=False) as d:references[key]=(d['truth'],d['harmonic'])
    rows=[];totals=dict(control_missed=0,recovered=0,near_existing_control_2um=0,far_from_control_7um=0,with_neighbor_candidate_5um=0)
    for key,prediction in predictions.items():
        truth,harmonic=references[key];old=matched_truth(harmonic,truth);new=pairs(prediction,truth)
        assert set(new)==matched_truth(prediction,truth)
        recovered=set(new)-old;totals['control_missed']+=len(truth)-len(old)
        for idx in sorted(recovered):
            point=prediction[new[idx]]
            nearest=float(cdist(point[None]*SCALE,harmonic*SCALE).min()) if len(harmonic) else None
            adjacent=[]
            for delta in [-1,1]:
                other=predictions.get((key[0],key[1]+delta))
                if other is not None and len(other):adjacent.append(float(cdist(point[None]*SCALE,other*SCALE).min()))
            row=dict(video=key[0],frame=key[1],truth_index=idx,candidate_to_control_um=nearest,adjacent_candidate_distance_um=adjacent,has_neighbor_within_5um=any(d<=5 for d in adjacent))
            rows.append(row);totals['recovered']+=1;totals['near_existing_control_2um']+=nearest is not None and nearest<=2;totals['far_from_control_7um']+=nearest is None or nearest>7;totals['with_neighbor_candidate_5um']+=row['has_neighbor_within_5um']
    receipt=json.loads((ROOT/'results/E027_completed.json').read_text());assert totals['recovered']==receipt['candidate_recovers_control_misses'] and totals['control_missed']==receipt['control_missed_centers']
    result=dict(status='complete',totals=totals,recoveries=rows,scope='Post-hoc calibration diagnosis; nearest neighbors do not establish cell identity, valid edges, divisions or precision',gpu=False,leaderboard_submitted=False)
    (ROOT/'results/E027_recovery_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
