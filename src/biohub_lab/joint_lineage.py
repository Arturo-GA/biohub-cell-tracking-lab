"""Joint selection of division trajectories and centers over temporal windows.

E010 is an original event-hypothesis optimizer. It uses the frozen Harmonic
graph as a reference and unselected E008/E009 centers as alternatives. A
division replaces a short primary chain and an orphan prefix together,
reconnecting two persistent anchors. No annotations are read by this module.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from .detector_proposals import SCALE, nms


@dataclass(frozen=True)
class JointConfig:
    horizon: int = 4
    persistence: int = 2
    proposal_nms_um: float = 1.2
    immutable_exclusion_um: float = 1.5
    search_um: float = 8.
    parent_um: float = 14.
    step_um: float = 6.
    min_sister_um: float = 2.6
    max_sister_um: float = 14.
    centroid_um: float = 4.
    candidates_per_frame: int = 6
    path_beam: int = 8
    max_orphans_per_mother: int = 4
    min_signal: float = .035
    max_pre_contrast: float = .10
    min_post_contrast: float = .06
    min_contrast_onset: float = .18
    min_mass_ratio: float = .5
    max_mass_ratio: float = 2.7
    milp_seconds: float = 60.


def graph_index(coords, edges):
    coords=np.asarray(coords);edges=np.asarray(edges,np.int64).reshape(-1,2)
    if not np.isfinite(coords).all() or np.any(coords!=np.rint(coords)):
        raise ValueError('Graph coordinates must be finite integers')
    children=defaultdict(list);parents={}
    for s,t in edges:
        if not 0<=s<len(coords) or not 0<=t<len(coords) or coords[t,0]!=coords[s,0]+1:
            raise ValueError('Invalid graph edge')
        if int(t) in parents:raise ValueError('Duplicate parent or edge')
        parents[int(t)]=int(s);children[int(s)].append(int(t))
        if len(children[int(s)])>2:raise ValueError('More than two children')
    return children,parents


def chain(start, length, children):
    values=[int(start)]
    for _ in range(length):
        following=children[values[-1]]
        if len(following)!=1:return None
        values.append(following[0])
    return values


def enumerate_windows(coords,edges,config=JointConfig()):
    """Anchor on persistent tracks; retain the old graph as the null hypothesis."""
    coords=np.asarray(coords);children,parents=graph_index(coords,edges)
    roots=defaultdict(list)
    for node in range(len(coords)):
        if node not in parents:roots[int(coords[node,0])].append(node)
    windows=[];h=config.horizon
    for mother in range(len(coords)):
        t=int(coords[mother,0])
        if t<2 or mother not in parents or parents[mother] not in parents:continue
        primary=chain(mother,h+config.persistence,children)
        if primary is None:continue
        choices=[]
        for birth in range(t+1,t+h+1):
            for orphan in roots[birth]:
                secondary=chain(orphan,t+h+config.persistence-birth,children)
                if secondary is None:continue
                anchor=secondary[t+h-birth]
                if np.linalg.norm((coords[orphan,1:]-coords[mother,1:])*SCALE)>config.parent_um:continue
                if np.linalg.norm((coords[anchor,1:]-coords[primary[h],1:])*SCALE)>config.max_sister_um:continue
                if set(primary)&set(secondary):continue
                distance=float(np.linalg.norm((coords[anchor,1:]-coords[mother,1:])*SCALE))
                choices.append((distance,orphan,secondary,birth))
        for _,orphan,secondary,birth in sorted(choices)[:config.max_orphans_per_mother]:
            removed=primary[1:h]+secondary[:t+h-birth]
            windows.append(dict(mother=mother,t=t,primary=primary,secondary=secondary,
                orphan=orphan,birth=birth,anchors=[primary[h],secondary[t+h-birth]],
                remove=removed,resources=sorted(set(primary+secondary+[parents[mother],parents[parents[mother]]]))))
    return windows


def combine_proposals(base, sources, shape, config=JointConfig()):
    """Merge detector alternatives without committing to their survival."""
    base=np.asarray(base,np.float32);points=[base];scores=[np.ones(len(base))];origin=[np.zeros(len(base),np.int8)]
    extras=[];confidence=[];provenance=[]
    for source_id,(coords,value) in enumerate(sources,1):
        coords=np.asarray(coords,np.float32);value=np.asarray(value,np.float32)
        if coords.shape!=(len(value),4) or not np.isfinite(coords).all() or not np.isfinite(value).all():
            raise ValueError('Malformed proposal source')
        if np.any(coords<0) or np.any(coords>np.asarray(shape)-1):raise ValueError('Out-of-volume proposal')
        if np.any(coords[:,0]!=np.rint(coords[:,0])):raise ValueError('Noninteger proposal time')
        extras.append(np.rint(coords));confidence.append(value);provenance.append(np.full(len(coords),source_id,np.int8))
    if extras:
        extras=np.concatenate(extras);confidence=np.concatenate(confidence);provenance=np.concatenate(provenance)
        for t in range(shape[0]):
            ids=np.flatnonzero(extras[:,0]==t)
            keep=ids[nms(extras[ids,1:],confidence[ids],config.proposal_nms_um)]
            # Exact native-coordinate duplicates use the original graph node.
            occupied={tuple(p[1:]) for p in base[base[:,0]==t]}
            keep=np.array([i for i in keep if tuple(extras[i,1:]) not in occupied],np.int64)
            points.append(extras[keep]);scores.append(confidence[keep]);origin.append(provenance[keep])
    return np.concatenate(points),np.concatenate(scores),np.concatenate(origin)


class ImageEvidence:
    """Locally background-subtracted fluorescence at half XY resolution."""
    spacing=SCALE*np.array([1,2,2])

    def __init__(self, volume):
        self.volume=np.asarray(volume,np.float32)
        if self.volume.ndim!=4 or not np.isfinite(self.volume).all():raise ValueError('Invalid image signal')

    @classmethod
    def from_zarr(cls,path):
        import zarr
        image=zarr.open_group(str(path),mode='r')['0']
        volume=np.empty((image.shape[0],image.shape[1],(image.shape[2]+1)//2,(image.shape[3]+1)//2),np.float32)
        for t in range(image.shape[0]):
            raw=np.asarray(image[t,:,::2,::2],np.float32)
            low,high=np.quantile(raw[::2,::2,::2],[.01,.999])
            frame=np.clip((raw-low)/max(float(high-low),1.),0,1)
            volume[t]=np.maximum(gaussian_filter(frame,.8/cls.spacing)-gaussian_filter(frame,5./cls.spacing),0)
        return cls(volume)

    def sample(self,t,physical):
        physical=np.asarray(physical).reshape(-1,3)
        return map_coordinates(self.volume[int(t)],(physical/self.spacing).T,order=1,mode='constant',cval=0.)

    def mass(self,t,center):
        offsets=np.array(list(product((-2.4,0.,2.4),repeat=3)))
        weights=np.exp(-np.sum(offsets**2,axis=1)/(2*2.4**2))
        return float(np.sum(self.sample(t,np.asarray(center)+offsets)*weights)/weights.sum())

    def contrast(self,t,a,b):
        values=self.sample(t,np.array([a,b,(a+b)/2]))
        value=float((values[0]+values[1]-2*values[2])/(values[0]+values[1]+2*values[2]+1.e-6))
        return value,values


def path_candidates(window,anchor_index,pool,confidence,trees,base_len,evidence,config,audit=None):
    """Beam search over alternative centers; the persistent endpoint is fixed."""
    coords=pool
    h=config.horizon;t=window['t'];mother=coords[window['mother'],1:]*SCALE
    anchor=coords[anchor_index,1:]*SCALE
    editable=set(window['remove']);allowed=editable|set(window['anchors'])
    base_frame={k:np.flatnonzero((coords[:base_len,0]==k)) for k in range(t+1,t+h)}
    beam=[(0.,[],mother)]
    for step in range(1,h):
        frame=t+step;expected=mother+(anchor-mother)*(step/h)
        ids,tree=trees[frame]
        possible=ids[tree.query_ball_point(expected,config.search_um)]
        stage=dict(frame=frame,inside_search=len(possible))
        possible=np.array([i for i in possible if i>=base_len or i in allowed],np.int64)
        stage['editable_or_proposed']=len(possible)
        immutable=np.array([i for i in base_frame[frame] if i not in editable],np.int64)
        if len(possible) and len(immutable):
            distances=cKDTree(coords[immutable,1:]*SCALE).query(coords[possible,1:]*SCALE)[0]
            possible=possible[(possible<base_len)|(distances>=config.immutable_exclusion_um)]
        stage['after_fixed_center_exclusion']=len(possible)
        if audit is not None:audit.append(stage)
        if not len(possible):return []
        physical=coords[possible,1:]*SCALE;signal=evidence.sample(frame,physical)
        rank=3*signal+.05*confidence[possible]-np.linalg.norm(physical-expected,axis=1)/config.search_um
        possible=possible[np.argsort(-rank,kind='stable')[:config.candidates_per_frame]]
        stage['ranked_candidates']=possible.tolist()
        expanded=[]
        for score,path,previous in beam:
            for node in possible:
                point=coords[node,1:]*SCALE;distance=np.linalg.norm(point-previous)
                if distance>(config.parent_um if step==1 else config.step_um):continue
                if np.linalg.norm(point-anchor)>(h-step)*config.step_um:continue
                value=float(evidence.sample(frame,point)[0])
                penalty=.01*distance**2 if step==1 else .025*distance**2
                if len(path)>=2:
                    previous_velocity=(coords[path[-1],1:]-coords[path[-2],1:])*SCALE
                    penalty+=.04*np.sum((point-previous-previous_velocity)**2)
                expanded.append((score+value-penalty,path+[int(node)],point))
        expanded.sort(key=lambda x:(-x[0],tuple(x[1])))
        beam=expanded[:config.path_beam]
        stage['feasible_extensions']=len(expanded);stage['retained_paths']=len(beam)
        if not beam:return []
    return [(score,path+[int(anchor_index)]) for score,path,_ in beam]


def score_pair(window,a,b,pool,evidence,config):
    """Compare appearance before/after the proposed split, not just edge distance."""
    physical_a=pool[a,1:]*SCALE;physical_b=pool[b,1:]*SCALE
    separation=np.linalg.norm(physical_a-physical_b,axis=1)
    if np.any(separation<config.min_sister_um) or np.any(separation>config.max_sister_um):return None,'separation'
    mother=pool[window['mother'],1:]*SCALE
    if np.linalg.norm((physical_a[0]+physical_b[0])/2-mother)>config.centroid_um:return None,'centroid'
    # Extrapolate each persistent daughter backwards using post-division motion.
    velocity_a=np.mean(np.diff(physical_a,axis=0),axis=0)
    velocity_b=np.mean(np.diff(physical_b,axis=0),axis=0)
    pre=[];pre_mid=[];post=[];post_peak=[];mass=[]
    for lag in (2,1):
        contrast,values=evidence.contrast(window['t']+1-lag,physical_a[0]-lag*velocity_a,physical_b[0]-lag*velocity_b)
        pre.append(contrast);pre_mid.append(float(values[2]))
    for step,(pa,pb) in enumerate(zip(physical_a,physical_b),1):
        contrast,values=evidence.contrast(window['t']+step,pa,pb)
        post.append(contrast);post_peak.append(float(min(values[:2])))
    pre_mean=float(np.mean(pre));post_median=float(np.median(post));onset=post_median-pre_mean
    if min(pre_mid)<config.min_signal:return None,'no_mother_image_signal'
    if min(post_peak)<config.min_signal:return None,'weak_daughter_image_signal'
    if pre_mean>config.max_pre_contrast:return None,'already_two_image_peaks'
    # Both daughters must be resolved from the first claimed daughter frame.
    # A median alone lets a path start at the old merged center and manufacture
    # an onset by moving to a pre-existing second nucleus one frame later.
    if min(post)<config.min_post_contrast or onset<config.min_contrast_onset:return None,'no_image_split_onset'
    for step,(pa,pb) in enumerate(zip(physical_a,physical_b),1):
        mass.append(evidence.mass(window['t']+step,pa)+evidence.mass(window['t']+step,pb))
    mother_mass=evidence.mass(window['t'],mother)
    ratio=float(np.median(mass)/max(mother_mass,1.e-6))
    if not config.min_mass_ratio<=ratio<=config.max_mass_ratio:return None,'mass_proxy'
    motion=float(np.mean(np.r_[np.sum(np.diff(physical_a,axis=0)**2,axis=1),np.sum(np.diff(physical_b,axis=0)**2,axis=1)]))
    gain=3*(onset-config.min_contrast_onset)+1.5*(post_median-config.min_post_contrast)+.5*max(0.,1-abs(np.log(ratio)))-.04*motion-.15
    evidence_record=dict(pre_contrast=pre,post_contrast=post,contrast_onset=onset,mass_proxy_ratio=ratio,
        mean_squared_step_um=motion,min_daughter_signal=min(post_peak),min_post_contrast=min(post),gain=float(gain))
    return evidence_record,'positive' if gain>0 else 'nonpositive_gain'


def solve_event_packing(events,pool,config=JointConfig()):
    """One choice per shared graph resource, plus spatial exclusion of new nodes."""
    if not events:return [],dict(status='empty',events=0,optimal=True)
    gains=np.array([e['evidence']['gain'] for e in events])
    if not np.isfinite(gains).all() or np.any(gains<=0):raise ValueError('Nonpositive event passed to packing')
    resource_events=defaultdict(set)
    for i,event in enumerate(events):
        for node in event['resources']:resource_events[('base',int(node))].add(i)
        for node in set(event['paths'][0]+event['paths'][1]):resource_events[('point',int(node))].add(i)
    conflicts=set()
    for owners in resource_events.values():
        for a,b in product(sorted(owners),repeat=2):
            if a<b:conflicts.add((a,b))
    # Different detector IDs can still describe the same physical cell.
    frame_points=defaultdict(list)
    for i,event in enumerate(events):
        for node in set(event['paths'][0][:-1]+event['paths'][1][:-1]):
            frame_points[int(pool[node,0])].append((i,node))
    for entries in frame_points.values():
        xyz=np.array([pool[n,1:]*SCALE for _,n in entries])
        for a,b in cKDTree(xyz).query_pairs(config.immutable_exclusion_um):
            ea,eb=entries[a][0],entries[b][0]
            if ea!=eb:conflicts.add(tuple(sorted((ea,eb))))
    if not conflicts:return list(range(len(events))),dict(status='independent',events=len(events),optimal=True)
    rr=[];cc=[]
    for row,(a,b) in enumerate(sorted(conflicts)):rr.extend([row,row]);cc.extend([a,b])
    matrix=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(conflicts),len(events))).tocsc()
    solution=milp(-gains,integrality=np.ones(len(events)),bounds=Bounds(0,1),
        constraints=LinearConstraint(matrix,0,1),options={'time_limit':config.milp_seconds,'mip_rel_gap':.001})
    if solution.x is None:raise RuntimeError('Joint optimizer returned no feasible selection: '+solution.message)
    chosen=np.rint(solution.x)
    if np.max(np.abs(chosen-solution.x))>1.e-5 or np.any(matrix@chosen>1.00001):raise ValueError('Infeasible joint selection')
    return np.flatnonzero(chosen).tolist(),dict(status=int(solution.status),events=len(events),
        conflicts=len(conflicts),gap=float(solution.mip_gap),optimal=bool(solution.status==0),message=solution.message)


def apply_events(base,edges,pool,events,selected):
    removed=set();resources=set();extra_edges=[];active_points=set()
    for index in selected:
        event=events[index]
        if resources.intersection(event['resources']):raise ValueError('Conflicting event resources')
        resources.update(event['resources']);removed.update(event['remove'])
        for path in event['paths']:
            full=[event['mother']]+path
            extra_edges.extend(zip(full[:-1],full[1:]));active_points.update(full)
    # A removed prefix vertex may be reselected as a center, but its old links
    # are replaced by the jointly selected paths.
    kept=(set(range(len(base)))-removed)|active_points
    kept=sorted(kept);lookup={old:new for new,old in enumerate(kept)}
    old_edges={(int(s),int(t)) for s,t in edges if s not in removed and t not in removed}
    # If an orphan starts exactly at the endpoint, there is no prefix to remove.
    for index in selected:
        event=events[index]
        old_edges={(s,t) for s,t in old_edges if s!=event['mother']}
    selected_edges=sorted(old_edges|set(extra_edges))
    coords=np.rint(pool[kept]).astype(np.int64)
    linked=np.array([(lookup[s],lookup[t]) for s,t in selected_edges],np.int64).reshape(-1,2)
    graph_index(coords,linked)
    if len({tuple(row) for row in coords})!=len(coords):raise ValueError('Duplicate final centers')
    return coords,linked,dict(removed_base_nodes=len(set(range(len(base)))-set(kept)),
        added_nodes=sum(i>=len(base) for i in kept),removed_base_edges=len(set(map(tuple,edges))-set(selected_edges)),
        added_edges=len(set(selected_edges)-set(map(tuple,edges))))


def reconstruct(base,edges,sources,shape,evidence,output,config=JointConfig()):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    pool,confidence,origin=combine_proposals(base,sources,shape,config)
    windows=enumerate_windows(base,edges,config)
    trees={t:(np.flatnonzero(pool[:,0]==t),cKDTree(pool[pool[:,0]==t,1:]*SCALE)) for t in range(shape[0])}
    events=[];rejections=Counter();inventory=[];started=time.monotonic()
    for window_index,window in enumerate(windows):
        stages=[[],[]]
        paths=[path_candidates(window,a,pool,confidence,trees,len(base),evidence,config,stages[i]) for i,a in enumerate(window['anchors'])]
        accepted=[];reasons=Counter()
        for (_,a),(_,b) in product(*paths):
            if set(a)&set(b):reasons['same_center']+=1;continue
            details,reason=score_pair(window,a,b,pool,evidence,config);reasons[reason]+=1
            if details is not None and details['gain']>0:
                accepted.append(dict(window,paths=[a,b],evidence=details))
        rejections.update(reasons)
        accepted.sort(key=lambda e:-e['evidence']['gain'])
        # Retain several geometry alternatives for global conflicts.
        event_ids=list(range(len(events),len(events)+min(3,len(accepted))))
        events.extend(accepted[:3])
        inventory.append(dict(mother=window['mother'],t=window['t'],orphan=window['orphan'],
            paths_per_anchor=[len(p) for p in paths],path_stages=stages,reasons=dict(reasons),event_ids=event_ids))
        if (window_index+1)%500==0:
            print('JOINT_PROGRESS',output.name,window_index+1,len(windows),'hypotheses',len(events),
                  'seconds',round(time.monotonic()-started,1),flush=True)
    selected,solver=solve_event_packing(events,pool,config)
    coords,linked,changes=apply_events(base,edges,pool,events,selected)
    np.savez_compressed(output/'candidate_pool.npz',coords=pool,confidence=confidence,origin=origin)
    np.savez_compressed(output/'final_graph.npz',coords=coords,edges=linked)
    with (output/'window_audit.jsonl').open('w') as handle:
        for row in inventory:handle.write(json.dumps(row)+'\n')
    with (output/'event_audit.jsonl').open('w') as handle:
        for i,event in enumerate(events):handle.write(json.dumps(dict(event,event_id=i,selected=i in selected))+'\n')
    receipt=dict(config=asdict(config),candidate_centers=len(pool)-len(base),windows=len(windows),
        positive_hypotheses=len(events),selected_events=len(selected),pair_decisions=dict(rejections),
        solver=solver,changes=changes,final_nodes=len(coords),final_edges=len(linked),
        sources='Frozen Harmonic graph, raw-image CELLECT/Gaussian proposals, raw .zarr intensities only',
        annotations_read=False,topology_validated=True)
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return coords,linked,receipt
