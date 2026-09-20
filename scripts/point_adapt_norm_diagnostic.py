"""CPU-only paired normalization diagnostic; no optimization or predictions for submission."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np

def locate(suffix):
    paths=list(Path('/kaggle/input').rglob(suffix))
    assert len(paths)==1,(suffix,len(paths))
    return paths[0]

def main(package):
    import tensorflow as tf
    assert not tf.config.list_physical_devices('GPU')
    start=time.monotonic();root=Path('/kaggle/working/normalization_diagnostic');root.mkdir(exist_ok=True)
    config=json.loads((package/'baseline/e025_point_adapt.json').read_text())
    data_path=locate('point_adapt_data/result.json');data=json.loads(data_path.read_text())
    train=json.loads(locate('point_adaptation/result.json').read_text())
    prep=json.loads(locate('nucverse_assets/result.json').read_text())
    assert data['config']==config
    selected=[]
    for split in ['fit','validation']:
        for prefix in ['44b6','6bba']:
            videos=sorted({r['video'] for r in data['crops'] if r['split']==split and r['video'].startswith(prefix)})[:2]
            for video in videos:selected.append(next(r for r in data['crops'] if r['split']==split and r['video']==video))
    sys.path.insert(0,str(package/'vendor/nucverse'))
    from attention_unet_3d import Attention_ResUNet_3D
    model=Attention_ResUNet_3D(tuple(config['patch'])+(1,),.2,True,32)
    # Both modes are deterministic: isolate batch statistics from dropout.
    dropout=[layer for layer in model.layers if isinstance(layer,tf.keras.layers.Dropout)]
    for layer in dropout:layer.rate=0.
    rows=[]
    for label,suffix,expected in [('original','nucverse_assets/best_model.weights.h5',prep['weight_sha256']),('adapted','point_adaptation/adapted.weights.h5',train['weight_sha256'])]:
        weight=locate(suffix);assert hashlib.sha256(weight.read_bytes()).hexdigest()==expected
        model.load_weights(weight)
        state=[v.numpy().copy() for v in model.non_trainable_variables]
        for item in selected:
            path=data_path.parent/item['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
            with np.load(path,allow_pickle=False) as d:
                image=d['image'][None,...,None].astype(np.float32);positive=d['positive'].astype(bool);negative=d['negative'].astype(bool)
            for batch_statistics in [False,True]:
                for variable,value in zip(model.non_trainable_variables,state):variable.assign(value)
                p=np.clip(model(image,training=batch_statistics)[0].numpy()[0,...,1],1e-6,1-1e-6)
                row=dict(model=label,video=item['video'],split=item['split'],mode='batch_statistics' if batch_statistics else 'stored_statistics',positive_probability=float(p[positive].mean()),positive_bce=float(-np.log(p[positive]).mean()),negative_probability=float(p[negative].mean()) if negative.any() else None)
                rows.append(row);print('NORMALIZATION',json.dumps(row),flush=True)
    groups=[]
    for label in ['original','adapted']:
        for split in ['fit','validation']:
            for mode in ['stored_statistics','batch_statistics']:
                subset=[r for r in rows if r['model']==label and r['split']==split and r['mode']==mode]
                groups.append(dict(model=label,split=split,mode=mode,n=len(subset),positive_bce=float(np.mean([r['positive_bce'] for r in subset])),positive_probability=float(np.mean([r['positive_probability'] for r in subset]))))
    result=dict(status='complete',gpu=False,optimizer_steps=0,dropout_disabled=len(dropout),state_restored_before_each_forward=True,selection='First crop of first two sorted videos per specimen and split',rows=rows,groups=groups,seconds=time.monotonic()-start,scope='Small paired forward-mode diagnostic, not tracking validation or a deployable improvement')
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('NORMALIZATION_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main(Path(sys.argv[1]))
