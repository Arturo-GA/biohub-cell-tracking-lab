"""Freeze a two-frame independent architecture comparison and split neural/CPU jobs."""
import base64,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    receipt=json.loads((ROOT/'results/E028_PREP_completed.json').read_text());assert receipt['status']=='complete'
    (ROOT/'baseline/e028_runtime.json').write_text(json.dumps(dict(manifest_sha256=receipt['result_sha256'],weight_sha256=receipt['weight_sha256']),indent=2)+'\n')
    for stage,gpu in [('inference',True),('evaluation',False)]:
        runner='scripts/cellpose_'+('inference' if gpu else 'evaluate')+'_runner.py'
        files=[runner,'scripts/cellpose_runtime.py','baseline/e028_runtime.json','baseline/e028_pilot.json','src/biohub_lab/__init__.py','src/biohub_lab/nucverse_instances.py']
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
            for name in files:
                data=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
                if name.endswith('.py'):compile(data,name,'exec')
                archive.writestr(name,data)
        payload=stream.getvalue();digest=hashlib.sha256(payload).hexdigest();folder=ROOT/('kaggle/cellpose_'+stage);folder.mkdir(exist_ok=True)
        code=f'''import base64,hashlib,io,zipfile,subprocess,sys,os
from pathlib import Path
payload=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={digest!r}
root=Path('/kaggle/working/cellpose_{stage}_package');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:archive.extractall(root)
subprocess.run([sys.executable,'-u',str(root/{runner!r}),str(root)],env=dict(os.environ,PYTHONPATH=str(root/'src')),check=True)
'''
        title='Biohub Lab Cellpose '+('Inference' if gpu else 'Evaluation CPU');slug='biohub-lab-cellpose-'+('inference' if gpu else 'evaluation-cpu')
        kernels=['jarturo/biohub-lab-cellpose-preparation-cpu','jarturo/biohub-lab-error-window-inputs-cpu']
        if not gpu:kernels.append('jarturo/biohub-lab-cellpose-inference')
        meta=dict(id='jarturo/'+slug,title=title,code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=gpu,enable_tpu=False,enable_internet=False,competition_sources=[],dataset_sources=[],kernel_sources=kernels,model_sources=[])
        if gpu:meta['machine_shape']='NvidiaTeslaT4'
        nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=[dict(cell_type='markdown',metadata={},source=['# E028 Cellpose-DINO '+stage+'\nTwo preselected full frames with control errors. '+('GPU neural fields only, no labels or training.' if gpu else 'CPU native mask decoder and sparse reference matching, no training.')+' No submission.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))])
        for name,value in [('notebook.ipynb',nb),('kernel-metadata.json',meta),('payload.json',dict(sha256=digest,files=files,contains_images=False,contains_weights=False,contains_credentials=False))]:(folder/name).write_text(json.dumps(value,indent=2)+'\n')
        checks=dict(payload_sha256=digest,notebook_sha256=hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest(),files=len(files),gpu=gpu,private=True,frames=2,gpu_justification='Neural inference through a different 86M-parameter transformer on two full volumes; preparation and masks/evaluation remain CPU')
        (ROOT/('results/E028_'+('GPU' if gpu else 'EVAL')+'_preflight.json')).write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':main()
