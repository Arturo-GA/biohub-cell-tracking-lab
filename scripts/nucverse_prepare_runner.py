"""E024 CPU: fetch one public generalist checkpoint and verify strict loading."""
import hashlib,json,os,struct,sys,time,traceback,zlib
from pathlib import Path

def main(package):
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    root=Path('/kaggle/working/nucverse_assets');root.mkdir(exist_ok=True)
    start=time.monotonic();record=dict(experiment='E024-preparation',status='downloading',gpu=False,competition_data_read=False)
    def save():(root/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        import requests
        pin=json.loads((package/'baseline/e024_nucverse.json').read_text())
        url=pin['weights_url'];offset=pin['member']['offset'];info=pin['member']
        response=requests.get(url,headers={'Range':f'bytes={offset}-{offset+1023}'},timeout=60)
        response.raise_for_status();assert response.status_code==206
        header=response.content;assert header[:4]==b'PK\x03\x04'
        method=struct.unpack_from('<H',header,8)[0];assert method==8
        namesize,extrasize=struct.unpack_from('<HH',header,26)
        assert header[30:30+namesize].decode()==info['name']
        begin=offset+30+namesize+extrasize;end=begin+info['compressed']-1
        target=root/'best_model.weights.h5';digest=hashlib.sha256();crc=0;size=0;received=0
        decoder=zlib.decompressobj(-15)
        with requests.get(url,headers={'Range':f'bytes={begin}-{end}'},stream=True,timeout=(30,60)) as response:
            response.raise_for_status();assert response.status_code==206
            assert response.headers['Content-Range'].startswith(f'bytes {begin}-{end}/')
            with target.open('wb') as handle:
                for block in response.iter_content(8*1024*1024):
                    received+=len(block);data=decoder.decompress(block);handle.write(data)
                    digest.update(data);crc=zlib.crc32(data,crc);size+=len(data)
                    print('WEIGHT_BYTES',received,info['compressed'],flush=True)
                data=decoder.flush();handle.write(data);digest.update(data);crc=zlib.crc32(data,crc);size+=len(data)
        assert decoder.eof and not decoder.unused_data and size==info['size'] and received==info['compressed'] and crc==info['crc']
        record.update(status='checking_model',weight_sha256=digest.hexdigest(),weight_bytes=size,source=pin)
        save()
        import tensorflow as tf
        import numpy as np
        assert not tf.config.list_physical_devices('GPU')
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        sys.path.insert(0,str(package/'vendor/nucverse'))
        from attention_unet_3d import Attention_ResUNet_3D
        tf.keras.backend.clear_session()
        # Spatial size changes only this functional smoke input, not convolution weights.
        model=Attention_ResUNet_3D((16,32,32,1),.2,True,32)
        model.load_weights(target)
        rng=np.random.default_rng(24);x=rng.random((1,16,32,32,1),dtype=np.float32)
        outputs=[v.numpy() for v in model(x,training=False)]
        assert [v.shape for v in outputs]==[(1,16,32,32,2),(1,16,32,32,3)]
        assert all(np.isfinite(v).all() for v in outputs)
        assert np.allclose(outputs[0].sum(-1),1,atol=1e-5)
        record.update(status='complete',tensorflow=tf.__version__,keras=tf.keras.__version__,numpy=np.__version__,
            parameters=model.count_params(),strict_weight_load=True,finite_smoke_outputs=True,
            seconds=time.monotonic()-start,scope='Compatibility only; no Biohub accuracy claim')
        save();print('NUCVERSE_PREPARED',json.dumps(record),flush=True)
    except Exception as error:
        record.update(status='failed',error=str(error));save();(root/'error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main(Path(sys.argv[1]))
