"""Hengck23 pretrained linker adapter; never loads annotations for inference."""
import hashlib
from pathlib import Path
import numpy as np
import torch


def load_independent_model(definitions, checkpoint, device='cuda'):
    scope={};exec(compile(definitions,'hengck23_model_v12_definitions.py','exec'),scope)
    D=scope['DotDict']
    cfg=D(unet_cfg=D(channel=(64,128,256),dropout=.1,gradient_checkpointing=False,
                    node_peak_kernel=3,node_peak_threshold=.2,max_detected_nodes=1024),
          tx_cfg=D(feat_dims=(64,128,256),hidden_dim=256,embed_dim=128,n_heads=4,n_layers=4,dropout=.1))
    model=scope['End2EndCellLinker'](cfg).to(device).eval()
    # The public checkpoint also stores its configuration as this audited
    # dict subclass. Allow only that definition, never unrestricted pickle.
    D.__module__='loss_and_metric_v12'
    with torch.serialization.safe_globals([D]):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(saved['model_state_dict'],strict=True)
    return model,scope['sample_pyr_feature_at_zyx'],hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()


@torch.inference_mode()
def sample_frame(model, sampler, volume, ids, nodes, device='cuda'):
    coords=np.array([[nodes[k][a] for a in ('z','y','x')] for k in ids],np.float32)/np.array([1,4,4],np.float32)
    xyz=torch.as_tensor(coords,device=device)[None]
    mask=torch.ones((1,len(ids)),dtype=torch.bool,device=device)
    v=torch.as_tensor(np.ascontiguousarray(volume),device=device)[None,None]
    with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=str(device).startswith('cuda')):
        (e0,e1,e2,d0,d1),_=model.unet.make_feature(v)
        feature=sampler([d0,d1,e2],xyz,v.shape[-3:],mask)
    return dict(ids=list(ids),zyx=xyz,mask=mask,feature=feature)


@torch.inference_mode()
def pair_logits(model, f0, f1, shape):
    with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=f0['zyx'].is_cuda):
        out=model.linker(f0['feature'],f1['feature'],f0['zyx'],f1['zyx'],f0['mask'],f1['mask'],shape)
    logits=out.edge_logit[0].float().cpu().numpy()
    assert np.isfinite(logits).all()
    return logits


def annotate_candidates(candidates, logits, ids0, ids1):
    r={k:i for i,k in enumerate(ids0)};c={k:i for i,k in enumerate(ids1)}
    rbest=logits.argmax(axis=1);cbest=logits.argmax(axis=0)
    result=[]
    for candidate in candidates:
        x=dict(candidate);a,b=r[x['a']],r[x['b']];da,db=c[x['da']],c[x['db']]
        aa,ab,ba,bb=(float(logits[i,j]) for i,j in ((a,da),(a,db),(b,da),(b,db)))
        x.update(old_logits=[aa,bb],new_logits=[ab,ba],new_min_logit=min(ab,ba),
                 min_row_gain=min(ab-aa,ba-bb),min_col_gain=min(ab-bb,ba-aa),
                 mutual_new=bool(rbest[a]==db and rbest[b]==da and cbest[db]==a and cbest[da]==b))
        result.append(x)
    return result


def annotate_joins(candidates,logits,ids0,ids1):
    r={k:i for i,k in enumerate(ids0)};c={k:i for i,k in enumerate(ids1)}
    result=[]
    for x in candidates:
        i,j=r[x['s']],c[x['d']];value=float(logits[i,j])
        # A zero-logit no-link alternative competes in both directions.
        row_alt=max(0.,float(np.max(np.delete(logits[i],j),initial=-np.inf)))
        col_alt=max(0.,float(np.max(np.delete(logits[:,j],i),initial=-np.inf)))
        result.append(dict(x,logit=value,row_margin=value-row_alt,col_margin=value-col_alt,
                           mutual=bool(logits[i].argmax()==j and logits[:,j].argmax()==i)))
    return result
