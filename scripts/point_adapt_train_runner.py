"""E025: real-image partial-label adaptation, then frozen full-frame inference."""
import gc,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
from biohub_lab.nucverse_tiles import predict_volume

def locate(suffix):
    paths=list(Path('/kaggle/input').rglob(suffix))
    if len(paths)!=1:raise ValueError(f'Expected one {suffix}; found {len(paths)}')
    return paths[0]

def main(package):
    root=Path('/kaggle/working/point_adaptation');root.mkdir(exist_ok=True);start=time.monotonic()
    record=dict(status='starting',experiment='E025',leaderboard_submitted=False)
    def save():(root/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        config=json.loads((package/'baseline/e025_point_adapt.json').read_text())
        data_path=locate('point_adapt_data/result.json');data=json.loads(data_path.read_text())
        assert data['status']=='complete' and data['config']==config and not set(config['fit'])&set(config['validation'])
        assert {r['video'] for r in data['crops'] if r['split']=='fit'}==set(config['fit'])
        assert {r['video'] for r in data['crops'] if r['split']=='validation'}==set(config['validation'])
        asset=locate('nucverse_assets/result.json');prep=json.loads(asset.read_text());weight=asset.parent/'best_model.weights.h5'
        pin=json.loads((package/'baseline/e024_runtime.json').read_text())
        assert hashlib.sha256(weight.read_bytes()).hexdigest()==pin['weight_sha256']==prep['weight_sha256']
        import tensorflow as tf
        gpus=tf.config.list_physical_devices('GPU');assert gpus
        for device in gpus:tf.config.experimental.set_memory_growth(device,True)
        tf.keras.utils.set_random_seed(config['seed'])
        sys.path.insert(0,str(package/'vendor/nucverse'))
        from attention_unet_3d import Attention_ResUNet_3D
        shape=tuple(config['patch'])+(1,)
        teacher=Attention_ResUNet_3D(shape,.2,True,32);teacher.load_weights(weight);teacher.trainable=False
        student=Attention_ResUNet_3D(shape,.2,True,32);student.load_weights(weight)
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        weights=config['weights']
        def average(value,mask):return tf.reduce_sum(value*mask)/tf.maximum(tf.reduce_sum(mask),1.)
        def objective(outputs,prior,positive,negative,flow_mask,target):
            p=tf.clip_by_value(tf.cast(outputs[0][...,1],tf.float32),1e-6,1-1e-6)
            q=tf.clip_by_value(tf.stop_gradient(tf.cast(prior[0][...,1],tf.float32)),1e-6,1-1e-6)
            unknown=1-tf.maximum(positive,negative)
            positive_loss=average(-tf.math.log(p),positive)
            negative_loss=average(-tf.math.log(1-p),negative)
            kl=q*tf.math.log(q/p)+(1-q)*tf.math.log((1-q)/(1-p))
            difference=tf.abs(tf.cast(outputs[1],tf.float32)-target)
            huber=tf.reduce_mean(tf.where(difference<1,.5*difference*difference,difference-.5),axis=-1)
            flow=average(huber,flow_mask)
            tether=average(tf.reduce_mean(tf.square(tf.cast(outputs[1],tf.float32)-tf.stop_gradient(tf.cast(prior[1],tf.float32))),axis=-1),unknown*tf.cast(q>.9,tf.float32))
            loss=weights['positive']*positive_loss+weights['dark']*negative_loss+weights['teacher_mask']*average(kl,unknown)+weights['flow']*flow+weights['teacher_flow']*tether
            return loss,tf.stack([positive_loss,negative_loss,flow,average(p,positive),average(p,negative)])
        @tf.function(reduce_retracing=True)
        def train_step(x,positive,negative,flow_mask,target):
            prior=teacher(x,training=False)
            with tf.GradientTape() as tape:
                outputs=student(x,training=True);loss,metrics=objective(outputs,prior,positive,negative,flow_mask,target)
            gradients=tape.gradient(loss,student.trainable_variables)
            tf.debugging.assert_all_finite(loss,'Non-finite loss')
            for gradient in gradients:tf.debugging.assert_all_finite(gradient,'Non-finite gradient')
            gradients,norm=tf.clip_by_global_norm(gradients,5.)
            optimizer.apply_gradients(zip(gradients,student.trainable_variables))
            return loss,metrics,norm
        @tf.function(reduce_retracing=True)
        def valid_step(x,positive,negative,flow_mask,target):
            prior=teacher(x,training=False)
            loss,metrics=objective(student(x,training=False),prior,positive,negative,flow_mask,target)
            return loss,metrics
        checked=set();rng=np.random.default_rng(config['seed'])
        def load(item,augment=False):
            path=data_path.parent/item['file']
            if str(path) not in checked:
                assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'];checked.add(str(path))
            with np.load(path,allow_pickle=False) as values:
                image=values['image'].astype(np.float32)
                if augment:image=np.clip(image*rng.uniform(.8,1.2),0,1)
                arrays=[image[None,...,None]]+[values[k][None].astype(np.float32) for k in ['positive','negative','flow_mask','flow']]
            return arrays
        fit=[r for r in data['crops'] if r['split']=='fit'];validation=[load(r) for r in data['crops'] if r['split']=='validation']
        history=[]
        def validate(step):
            values=[]
            for batch in validation:
                loss,metrics=valid_step(*batch);values.append([float(loss),*metrics.numpy().tolist()])
            row=dict(step=step,validation=np.mean(values,axis=0).tolist());history.append(row)
            (root/'history.json').write_text(json.dumps(history,indent=2)+'\n');print('POINT_VALIDATION',row,flush=True)
        validate(0)
        record.update(status='training',training_videos=config['fit'],validation_videos=config['validation'],steps=config['steps']);save()
        order=rng.permutation(len(fit));position=0
        for step in range(1,config['steps']+1):
            if time.monotonic()-start>config['gpu_training_deadline_seconds']:raise TimeoutError('Frozen training runtime budget exceeded')
            if position==len(order):order=rng.permutation(len(fit));position=0
            loss,metrics,norm=train_step(*load(fit[int(order[position])],True));position+=1
            if step%50==0:print('POINT_TRAIN',step,float(loss),float(norm),flush=True)
            if step%config['validate_every']==0:validate(step)
        target=root/'adapted.weights.h5';student.save_weights(target)
        adapted_sha=hashlib.sha256(target.read_bytes()).hexdigest()
        record.update(status='trained',training_seconds=time.monotonic()-start,weight_sha256=adapted_sha,history=history,
            tensorflow=tf.__version__,source_weight_sha256=pin['weight_sha256'],checkpoint_selection='Fixed final step',training_manifest_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest())
        save()
        # The full-frame benchmark reads no labels; reuse exactly E024's frozen images.
        del train_step,valid_step,student,teacher,optimizer,validation
        tf.keras.backend.clear_session();gc.collect()
        bench=json.loads((package/'baseline/e024_benchmark.json').read_text())
        model=Attention_ResUNet_3D(tuple(bench['patch'])+(1,),.2,True,32);model.load_weights(target)
        @tf.function(reduce_retracing=True)
        def forward(x):return model(x,training=False)
        inputs=locate('nucverse_inputs/result.json');manifest=json.loads(inputs.read_text());assert manifest['selection']==bench
        output=Path('/kaggle/working/nucverse_fields');output.mkdir(exist_ok=True);frames=[];inference_start=time.monotonic()
        for item in manifest['frames']:
            path=inputs.parent/item['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
            tick=time.monotonic();fields,tiling=predict_volume(forward,np.load(path,allow_pickle=False),tuple(bench['patch']),tuple(bench['stride']),inference_start+1800)
            prediction=output/(path.stem+'_fields.npy');np.save(prediction,fields.astype(np.float16),allow_pickle=False)
            frames.append(dict(**item,output=prediction.name,output_sha256=hashlib.sha256(prediction.read_bytes()).hexdigest(),seconds=time.monotonic()-tick,tiling=tiling))
        (output/'result.json').write_text(json.dumps(dict(status='complete',experiment='E025',annotations_read=False,training=True,frames=frames,seconds=time.monotonic()-inference_start,weight_sha256=adapted_sha),indent=2)+'\n')
        record.update(status='complete',total_seconds=time.monotonic()-start);save();print('POINT_ADAPT_COMPLETE',json.dumps(record),flush=True)
    except Exception as error:
        record.update(status='failed',error=str(error),seconds=time.monotonic()-start);save();(root/'error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main(Path(sys.argv[1]))
