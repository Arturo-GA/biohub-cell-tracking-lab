"""Build three private E030 stages: CPU data, GPU tokens, CPU learning."""
import base64,hashlib,io,json,zipfile
from pathlib import Path
from build_tissue_trajectory_notebooks import package as cpu_package
ROOT=Path(__file__).resolve().parents[1]


def main():
    split=json.loads((ROOT/'baseline/event_graph_split.json').read_text())['split']
    config=dict(experiment='E030',seed=300920,fit=split['fit'],validation=split['calibration'],
        transitions_per_video=8,jitter_um=1.,candidate_k=8,radius_um=20.,features=4608,hidden=96,
        steps=600,learning_rate=.001,feature_batch_size=64,gpu_deadline_seconds=2400,
        selection='Fixed final step. All fit division frames plus uniform fit transitions. Uniform calibration transitions, no control error selection.',
        scope='Annotated jitter curriculum; actual detector-space and official graph validation still required')
    (ROOT/'baseline/e030_image_identity.json').write_text(json.dumps(config,indent=2)+'\n')
    core=['src/biohub_lab/__init__.py','src/biohub_lab/image_identity.py','baseline/e030_image_identity.json']
    for stage in ('data','features','train'):
        gpu=stage=='features';folder=ROOT/('kaggle/image_identity_'+stage)
        runner='scripts/image_identity_'+stage+'_runner.py'
        files=core+[runner]
        kernels=[] if stage=='data' else ['jarturo/biohub-image-identity-data-cpu']
        if gpu:
            files+=['scripts/cellpose_runtime.py','baseline/e028_runtime.json'];kernels+=['jarturo/biohub-lab-cellpose-preparation-cpu']
        if stage=='train':
            kernels+=['jarturo/biohub-image-identity-features'];files+=['baseline/e030_head_protocol.json']
        slug='biohub-image-identity-'+stage+('' if gpu else '-cpu');title='Biohub Image Identity '+stage.title()+('' if gpu else ' CPU')
        if stage=='data':
            checks=cpu_package(Path('kaggle/identity_parent'),folder,files,'scripts/identity_parent_runner.py',runner,slug,title,False,kernels)
            nb=json.loads((folder/'notebook.ipynb').read_text());nb['cells'][0]['source']=['# E030 image identity data\nCPU real-image three-plane crops; separate fit and calibration videos. Jittered annotated centers, not detection validation.\n']
            (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
        else:
            stream=io.BytesIO()
            with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
                for name in files:
                    content=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
                    if name.endswith('.py'):compile(content,name,'exec')
                    archive.writestr(zipfile.ZipInfo(name,date_time=(2026,9,20,0,0,0)),content,compress_type=zipfile.ZIP_DEFLATED)
            payload=stream.getvalue();digest=hashlib.sha256(payload).hexdigest();folder.mkdir(exist_ok=True)
            code=f'''import base64,hashlib,io,zipfile,subprocess,sys,os
from pathlib import Path
payload=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={digest!r}
root=Path('/kaggle/working/image_identity_{stage}_package');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:archive.extractall(root)
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'src'),str(root/'scripts')]),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
if not {gpu!r}:env['CUDA_VISIBLE_DEVICES']=''
subprocess.run([sys.executable,'-u',str(root/{runner!r}),str(root)],env=env,check=True)
'''
            compile(code,'notebook','exec')
            nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=[dict(cell_type='markdown',metadata={},source=['# E030 '+title+'\nActual normalized DINO tokens, never random styles. Frozen checkpoint. CPU heads and geometry ablation; no automatic submission.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))])
            meta=dict(id='jarturo/'+slug,title=title,code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=gpu,enable_tpu=False,enable_internet=False,competition_sources=[],dataset_sources=[],kernel_sources=kernels,model_sources=[])
            if gpu:meta['machine_shape']='NvidiaTeslaT4'
            for name,value in [('notebook.ipynb',nb),('kernel-metadata.json',meta),('payload.json',dict(sha256=digest,files=files,contains_images=False,contains_weights=False,contains_credentials=False))]: (folder/name).write_text(json.dumps(value,indent=2)+'\n')
            checks=dict(payload_sha256=digest,files=len(files),gpu=gpu,private=True)
        checks.update(notebook_sha256=hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest(),stage=stage)
        (ROOT/('results/E030_'+stage.upper()+'_preflight.json')).write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))

if __name__=='__main__':main()
