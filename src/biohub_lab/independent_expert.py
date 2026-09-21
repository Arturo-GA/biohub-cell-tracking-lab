"""Public UNet/Transformer inference independent of the existing fusion gate.

Architecture is read from the pinned pilkwang support pack; only the reviewed
model and positional-encoding definitions are compiled from the training file.
"""
import ast,hashlib,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def architecture(repo):
    sys.path.insert(0,str(Path(repo)/'src'))
    from biohub_tracking.models import SimpleNodeTransformer,TemporalUNet3D
    source=(Path(repo)/'scripts/train_unet_transformer.py').read_text()
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('UNetNodeTransformer','extract_pos_features')]
    assert len(nodes)==2
    namespace=dict(torch=torch,nn=nn,F=F,np=np,SimpleNodeTransformer=SimpleNodeTransformer,_POS_EMBED_DIM=8)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'reviewed_public_model','exec'),namespace)
    model=namespace['UNetNodeTransformer'](TemporalUNet3D(in_channels=1,out_channels=32,layers=[32,64,128]),32,32)
    return model,namespace['extract_pos_features']

def load(repo,path,digest,device):
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    model,pos=architecture(repo);state=torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True);model.to(device).eval()
    return model,pos

def probabilities(model,pos,features,source,target):
    """Coordinates are native voxels from the frozen detection cache."""
    device=features.device;down=np.array([1,4,4],np.float32);shape=(2,*features.shape[3:])
    c=[];coords=[];masks=[];positions=[]
    for i,points in enumerate([source,target]):
        v=np.asarray(points,dtype=np.float32).copy();v[:,0]=i;v[:,1:]/=down;c.append(v)
        coords.append(torch.as_tensor(v[:,1:],device=device)[None]);masks.append(torch.ones((1,len(v)),device=device,dtype=torch.bool))
        positions.append(torch.as_tensor(pos(v,shape),device=device)[None])
    if not len(source) or not len(target):return np.zeros((len(source),len(target)),np.float32)
    feat=[model._index_features(features[:,i],coords[i],masks[i]) for i in range(2)]
    d=torch.as_tensor(down,device=device)
    logits=model.predict_edges(*feat,coords[0]*d,coords[1]*d,*positions,*masks)
    return torch.softmax(logits[0].float(),dim=0).cpu().numpy()

def sparse(prob,old_edges,topk=8):
    keep=np.zeros(prob.shape,bool)
    if prob.size:
        k=min(topk,prob.shape[1]);keep[np.arange(prob.shape[0])[:,None],np.argpartition(prob,-k,axis=1)[:,-k:]]=True
        k=min(topk,prob.shape[0]);keep[np.argpartition(prob,-k,axis=0)[-k:,:],np.arange(prob.shape[1])[None,:]]=True
        old=np.asarray(old_edges,dtype=int).reshape(-1,2);keep[old[:,0],old[:,1]]=True
    a,b=np.nonzero(keep);return np.c_[a,b],prob[a,b]
