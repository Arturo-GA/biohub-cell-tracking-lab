import base64,hashlib,io,json,zipfile
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,package,sha

def main():
    config=dict(experiment='E033',seed=330920,steps=1000,batch=16,temperature=.2,learning_rate=.0003,deadline_seconds=1500,encoder='3-channel temporal Conv3D, GroupNorm,64-dimensional embedding; trained from scratch',split='Two directions, train on one embryo and predict the other',prediction_manifest_sha256=json.loads((ROOT/'results/E031_GPU_completed.json').read_text())['result_sha256'],image_manifest_sha256=json.loads((ROOT/'results/E031_PREP_completed.json').read_text())['result_sha256'],promotion='Aggregate accuracy at least0.005 above nearest, neither embryo below nearest, and0.005 above image-shuffled control; then full graph evaluation still required')
    (ROOT/'baseline/e033_temporal_volume.json').write_text(json.dumps(config,indent=2)+'\n')
    for stage in ['prepare','train','evaluate']:
        if stage=='prepare' and (ROOT/'results/E033_PREPARE_launch.json').exists():continue
        folder=ROOT/('kaggle/temporal_volume_'+stage);runner='scripts/temporal_volume_'+stage+'_runner.py'
        files=['baseline/e033_temporal_volume.json','src/biohub_lab/temporal_volume.py',runner]
        if stage=='train':
            prepared=ROOT/'outputs/e033_prepare/temporal_volume_data/result.json'
            if not prepared.exists():continue
            assert json.loads(prepared.read_text())['status']=='complete'
            (ROOT/'baseline/e033_input_pin.json').write_text(json.dumps(dict(manifest_sha256=sha(prepared)),indent=2)+'\n')
            files+=['baseline/e033_input_pin.json']
        if stage=='prepare':
            files+=json.loads((ROOT/'kaggle/dense_temporal_rank/payload.json').read_text())['files']
            checks=package(Path('kaggle/identity_parent'),folder,files,'scripts/identity_parent_runner.py',runner,'biohub-temporal-volume-prepare-cpu','Biohub Temporal Volume Prepare CPU',False,['jarturo/biohub-dense-detector-gpu','jarturo/biohub-dense-detector-prepare-cpu'])
            nb=json.loads((folder/'notebook.ipynb').read_text());nb['cells'][0]['source']=['# E033 temporal volumetric crops\nReal detector candidates, three time points per3D crop, CPU preparation.']
            (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n');checks['notebook_sha256']=sha(folder/'notebook.ipynb')
        else:
            if not (ROOT/runner).exists():continue
            # Candidate-generation dependencies are imported lazily; GPU requires torch/numpy only.
            stream=io.BytesIO()
            with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
                for name in files:
                    content=(ROOT/name).read_text(encoding='utf8')
                    if name.endswith('.py'):compile(content,name,'exec')
                    archive.writestr(name,content.encode('utf8'))
            payload=stream.getvalue();digest=hashlib.sha256(payload).hexdigest();folder.mkdir(exist_ok=True)
            code=f'''import base64,hashlib,io,zipfile,subprocess,sys,os
from pathlib import Path
payload=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={digest!r}
root=Path('/kaggle/working/temporal_volume_{stage}_package');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:archive.extractall(root)
env=dict(os.environ,PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
if {stage!='train'!r}:env['CUDA_VISIBLE_DEVICES']=''
subprocess.run([sys.executable,'-u',str(root/{runner!r}),str(root)],env=env,check=True)
'''
            compile(code,'notebook','exec')
            nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python')),cells=[dict(cell_type='markdown',metadata={},source=['# E033 '+stage+'\nTrain actual volumetric encoder; separate CPU evaluation. No automatic submission.']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))])
            kernels=['jarturo/biohub-temporal-volume-prepare-cpu']
            if stage=='evaluate':kernels+=['jarturo/biohub-temporal-volume-train']
            meta=dict(id='jarturo/biohub-temporal-volume-'+stage+('' if stage=='train' else '-cpu'),title='Biohub Temporal Volume '+stage.title()+('' if stage=='train' else ' CPU'),code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=stage=='train',enable_tpu=False,enable_internet=False,competition_sources=[],dataset_sources=[],kernel_sources=kernels,model_sources=[])
            if stage=='train':meta['machine_shape']='NvidiaTeslaT4'
            for name,value in [('notebook.ipynb',nb),('kernel-metadata.json',meta),('payload.json',dict(sha256=digest,files=files,contains_credentials=False,contains_weights=False,contains_images=False))]:(folder/name).write_text(json.dumps(value,indent=2)+'\n')
            checks=dict(notebook_sha256=sha(folder/'notebook.ipynb'),payload_sha256=digest,gpu=stage=='train',private=True)
        (ROOT/('results/E033_'+stage.upper()+'_preflight.json')).write_text(json.dumps(checks,indent=2)+'\n');print(stage,checks)
if __name__=='__main__':main()
