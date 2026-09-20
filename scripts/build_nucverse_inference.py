"""Build GPU-only inference after CPU assets have passed strict loading."""
import base64,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    prep=json.loads((ROOT/'outputs/e024_prep_recovery/nucverse_assets/result.json').read_text());assert prep['status']=='complete'
    pin={k:prep[k] for k in ['weight_sha256','tensorflow','keras','numpy']}
    (ROOT/'baseline/e024_runtime.json').write_text(json.dumps(pin,indent=2)+'\n')
    names=['scripts/nucverse_inference_runner.py','src/biohub_lab/__init__.py','src/biohub_lab/nucverse_tiles.py','baseline/e024_runtime.json','baseline/e024_benchmark.json','vendor/nucverse/attention_unet_3d.py','vendor/nucverse/LICENSE']
    folder=ROOT/'kaggle/nucverse_inference';folder.mkdir(exist_ok=True);stream=io.BytesIO()
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
root=Path('/kaggle/working/nucverse_inference_package');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:archive.extractall(root)
subprocess.run([sys.executable,'-u',str(root/'scripts/nucverse_inference_runner.py'),str(root)],env=dict(os.environ,PYTHONPATH=str(root/'src')),check=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=[dict(cell_type='markdown',metadata={},source=['# E024 NucVerse full-frame inference\nEight fixed frames, generalist 3D segmenter. GPU for neural fields only; CPU decoding and evaluation follow. No training or submission.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))])
    meta=dict(id='jarturo/biohub-lab-nucverse-inference',title='Biohub Lab NucVerse Inference',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,machine_shape='NvidiaTeslaT4',competition_sources=[],dataset_sources=[],kernel_sources=['jarturo/biohub-lab-nucverse-preparation-cpu','jarturo/biohub-lab-nucverse-inputs-cpu'],model_sources=[])
    for name,value in [('notebook.ipynb',nb),('kernel-metadata.json',meta),('payload.json',dict(sha256=digest,files=names,contains_images=False,contains_weights=False,contains_credentials=False))]:
        (folder/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    receipt=dict(payload_sha256=digest,notebook_sha256=hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest(),files=len(names),gpu=True,private=True,frames=8,gpu_justification='Full-frame volumetric neural segmentation only; downloads and instance decoding/evaluation use CPU')
    (ROOT/'results/E024_GPU_preflight.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
