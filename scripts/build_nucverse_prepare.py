"""Build the private CPU-only public asset preparation notebook."""
import base64,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    folder=ROOT/'kaggle/nucverse_prepare';folder.mkdir(exist_ok=True)
    names=['scripts/nucverse_prepare_runner.py','baseline/e024_nucverse.json','vendor/nucverse/attention_unet_3d.py','vendor/nucverse/LICENSE']
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            data=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
            if name.endswith('.py'):compile(data,name,'exec')
            archive.writestr(name,data)
    payload=stream.getvalue();digest=hashlib.sha256(payload).hexdigest()
    code=f'''import base64,hashlib,io,zipfile,subprocess,sys
from pathlib import Path
payload=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={digest!r}
root=Path('/kaggle/working/nucverse_prepare_package');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:archive.extractall(root)
subprocess.run([sys.executable,'-u',str(root/'scripts/nucverse_prepare_runner.py'),str(root)],check=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=[dict(cell_type='markdown',metadata={},source=['# E024 NucVerse3D asset preparation\nCPU only. Public generalist weights; no competition images, no training, no submission.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))])
    meta=dict(id='jarturo/biohub-lab-nucverse-preparation-cpu',title='Biohub Lab NucVerse Preparation CPU',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=True,competition_sources=[],dataset_sources=[],kernel_sources=[],model_sources=[])
    for name,value in [('notebook.ipynb',nb),('kernel-metadata.json',meta),('payload.json',dict(sha256=digest,files=names,contains_images=False,contains_weights=False,contains_credentials=False))]:
        (folder/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(payload_sha256=digest,files=len(names),gpu=False),indent=2))
if __name__=='__main__':main()
