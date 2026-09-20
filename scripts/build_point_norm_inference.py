"""Package E026 deterministic image-local normalization with frozen adapted weights."""
import base64,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    names=['scripts/point_norm_inference_runner.py','src/biohub_lab/__init__.py','src/biohub_lab/nucverse_tiles.py','baseline/e026_normalization.json','baseline/e024_benchmark.json','vendor/nucverse/attention_unet_3d.py','vendor/nucverse/LICENSE','NOTICE.md']
    folder=ROOT/'kaggle/point_norm_inference';folder.mkdir(exist_ok=True);stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            data=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
            if name.endswith('.py'):compile(data,name,'exec')
            archive.writestr(name,data)
    payload=stream.getvalue();digest=hashlib.sha256(payload).hexdigest()
    code=f'''import base64,hashlib,io,zipfile,subprocess,sys,os
from pathlib import Path
payload=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={digest!r}
root=Path('/kaggle/working/point_norm_inference_package');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:archive.extractall(root)
subprocess.run([sys.executable,'-u',str(root/'scripts/point_norm_inference_runner.py'),str(root)],env=dict(os.environ,PYTHONPATH=str(root/'src')),check=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=[dict(cell_type='markdown',metadata={},source=['# E026 normalization correction\nFrozen E025 weights. Per-tile batch statistics with dropout disabled and moving statistics unchanged. No training, annotations or submission.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))])
    meta=dict(id='jarturo/biohub-lab-point-normalization-inference',title='Biohub Lab Point Normalization Inference',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,machine_shape='NvidiaTeslaT4',competition_sources=[],dataset_sources=[],kernel_sources=['jarturo/biohub-lab-point-adapt-training','jarturo/biohub-lab-nucverse-inputs-cpu'],model_sources=[])
    for name,value in [('notebook.ipynb',nb),('kernel-metadata.json',meta),('payload.json',dict(sha256=digest,files=names,contains_images=False,contains_weights=False,contains_credentials=False))]:
        (folder/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    receipt=dict(payload_sha256=digest,notebook_sha256=hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest(),files=len(names),gpu=True,private=True,optimizer_steps=0,gpu_justification='Neural inference only for eight full 3D frames; CPU diagnosis and evaluation')
    (ROOT/'results/E026_GPU_preflight.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
