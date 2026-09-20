"""Package the substantive point-supervised segmentation adaptation experiment."""
import base64,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    names=['scripts/point_adapt_train_runner.py','src/biohub_lab/__init__.py','src/biohub_lab/nucverse_tiles.py','baseline/e025_point_adapt.json','baseline/e024_runtime.json','baseline/e024_benchmark.json','vendor/nucverse/attention_unet_3d.py','vendor/nucverse/LICENSE','NOTICE.md']
    folder=ROOT/'kaggle/point_adapt_train';folder.mkdir(exist_ok=True);stream=io.BytesIO()
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
root=Path('/kaggle/working/point_adapt_train_package');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:archive.extractall(root)
subprocess.run([sys.executable,'-u',str(root/'scripts/point_adapt_train_runner.py'),str(root)],env=dict(os.environ,PYTHONPATH=str(root/'src')),check=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=[dict(cell_type='markdown',metadata={},source=['# E025 point-supervised segmentation adaptation\nReal-image training on 48 videos; 16 separate validation videos. Unknown voxels are not background labels. Fixed 800-step checkpoint, then frozen full-frame inference; no submission.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))])
    meta=dict(id='jarturo/biohub-lab-point-adapt-training',title='Biohub Lab Point Adapt Training',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,machine_shape='NvidiaTeslaT4',competition_sources=[],dataset_sources=[],kernel_sources=['jarturo/biohub-lab-nucverse-preparation-cpu','jarturo/biohub-lab-point-adapt-data-cpu','jarturo/biohub-lab-nucverse-inputs-cpu'],model_sources=[])
    for name,value in [('notebook.ipynb',nb),('kernel-metadata.json',meta),('payload.json',dict(sha256=digest,files=names,contains_images=False,contains_weights=False,contains_credentials=False))]:
        (folder/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    receipt=dict(payload_sha256=digest,notebook_sha256=hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest(),files=len(names),gpu=True,private=True,fit_videos=48,validation_videos=16,steps=800,gpu_justification='Backpropagation through a 40M-parameter volumetric segmenter and neural inference; data preparation and instance evaluation use CPU')
    (ROOT/'results/E025_TRAIN_preflight.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
