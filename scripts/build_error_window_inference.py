"""Package E027 deterministic image-local normalization with frozen adapted weights."""
import base64,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    names=['scripts/error_window_inference_runner.py','src/biohub_lab/__init__.py','src/biohub_lab/nucverse_tiles.py','baseline/e026_normalization.json','baseline/e027_error_windows.json','vendor/nucverse/attention_unet_3d.py','vendor/nucverse/LICENSE','NOTICE.md']
    folder=ROOT/'kaggle/error_window_inference';folder.mkdir(exist_ok=True);stream=io.BytesIO()
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
root=Path('/kaggle/working/error_window_inference_package');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:archive.extractall(root)
subprocess.run([sys.executable,'-u',str(root/'scripts/error_window_inference_runner.py'),str(root)],env=dict(os.environ,PYTHONPATH=str(root/'src')),check=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=[dict(cell_type='markdown',metadata={},source=['# E027 complementary detector on control errors\nFrozen E025 weights and E026 normalization. Full-frame image-only inference on error-conditioned calibration windows. No training or submission.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))])
    meta=dict(id='jarturo/biohub-lab-error-window-inference',title='Biohub Lab Error Window Inference',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,machine_shape='NvidiaTeslaT4',competition_sources=[],dataset_sources=[],kernel_sources=['jarturo/biohub-lab-point-adapt-training','jarturo/biohub-lab-error-window-inputs-cpu'],model_sources=[])
    for name,value in [('notebook.ipynb',nb),('kernel-metadata.json',meta),('payload.json',dict(sha256=digest,files=names,contains_images=False,contains_weights=False,contains_credentials=False))]:
        (folder/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    receipt=dict(payload_sha256=digest,notebook_sha256=hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest(),files=len(names),gpu=True,private=True,optimizer_steps=0,gpu_justification='Neural inference only for error-window full 3D frames; CPU diagnosis and evaluation')
    (ROOT/'results/E027_GPU_preflight.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
