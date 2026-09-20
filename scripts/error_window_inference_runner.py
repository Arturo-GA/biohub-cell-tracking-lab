"""E027 GPU: neural fields only; instance decoding/evaluation remain CPU."""
import hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
from biohub_lab.nucverse_tiles import predict_volume

def locate(suffix):
    paths=list(Path('/kaggle/input').rglob(suffix))
    if len(paths)!=1:raise ValueError(f'Expected one {suffix}; found {len(paths)}')
    return paths[0]

def main(package):
    root=Path('/kaggle/working/nucverse_fields');root.mkdir(exist_ok=True);start=time.monotonic()
    record=dict(status='starting',experiment='E027',annotations_read=False,training=False,leaderboard_submitted=False)
    def save():(root/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        asset=locate('point_adaptation/result.json');prep=json.loads(asset.read_text());assert prep['status']=='complete'
        pin=json.loads((package/'baseline/e026_normalization.json').read_text())
        weight=asset.parent/'adapted.weights.h5'
        assert hashlib.sha256(weight.read_bytes()).hexdigest()==pin['weight_sha256']==prep['weight_sha256']
        inputs=locate('nucverse_inputs/result.json');manifest=json.loads(inputs.read_text());assert manifest['status']=='complete'
        config=json.loads((package/'baseline/e027_error_windows.json').read_text());assert manifest['selection']==config
        import tensorflow as tf
        devices=tf.config.list_physical_devices('GPU');assert devices
        for device in devices:tf.config.experimental.set_memory_growth(device,True)
        sys.path.insert(0,str(package/'vendor/nucverse'))
        from attention_unet_3d import Attention_ResUNet_3D
        model=Attention_ResUNet_3D(tuple(config['patch'])+(1,),.2,True,32);model.load_weights(weight)
        batch_norm=[layer for layer in model.layers if isinstance(layer,tf.keras.layers.BatchNormalization)]
        dropout=[layer for layer in model.layers if isinstance(layer,tf.keras.layers.Dropout)]
        assert batch_norm and dropout
        for layer in batch_norm:layer.momentum=1.
        for layer in dropout:layer.rate=0.
        stored=[v.numpy().copy() for v in model.non_trainable_variables]
        @tf.function(reduce_retracing=True)
        def forward(x):return model(x,training=True)
        reports=[]
        for item in manifest['frames']:
            path=inputs.parent/item['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
            image=np.load(path,allow_pickle=False);tick=time.monotonic()
            fields,report=predict_volume(forward,image,tuple(config['patch']),tuple(config['stride']),start+config['gpu_deadline_seconds'])
            target=root/(path.stem+'_fields.npy');np.save(target,fields.astype(np.float16),allow_pickle=False)
            reports.append(dict(**item,output=target.name,output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),seconds=time.monotonic()-tick,tiling=report))
            print('NUCVERSE_FRAME',item['video'],item['frame'],reports[-1]['seconds'],flush=True)
        assert all(np.array_equal(before,variable.numpy()) for before,variable in zip(stored,model.non_trainable_variables))
        record.update(normalization='Per-tile batch statistics; batch size one; dropout disabled',moving_statistics_unchanged=True,optimizer_steps=0)
        record.update(status='complete',frames=reports,seconds=time.monotonic()-start,tensorflow=tf.__version__,keras=tf.keras.__version__,numpy=np.__version__,weight_sha256=pin['weight_sha256'])
        save();print('NUCVERSE_FIELDS_COMPLETE',json.dumps(record),flush=True)
    except Exception as error:
        record.update(status='failed',error=str(error),seconds=time.monotonic()-start);save();(root/'error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main(Path(sys.argv[1]))
