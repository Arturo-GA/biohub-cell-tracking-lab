"""Package a CPU-only Cellpose-DINO compatibility preparation, no competition data."""
import base64,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    config=dict(cellpose_version='4.2.1.1',cellpose_reference_commit='a54cb48849b7e225a81e8e43dcb042d42427f543',dinov3_commit='6876159a11b4df116f30f667f8c9888617df0751',hf_revision='7c61431b5fbb078f3296754bd15d9f51b320f837',model='cpdino-vitb',model_card_license='bsd-3-clause',sources=['https://cellpose.readthedocs.io/en/latest/models.html','https://huggingface.co/mouseland/cellpose-sam','https://github.com/facebookresearch/dinov3'])
    (ROOT/'baseline/e028_cellpose.json').write_text(json.dumps(config,indent=2)+'\n')
    files=['scripts/cellpose_prepare_runner.py','baseline/e028_cellpose.json'];stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            data=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
            if name.endswith('.py'):compile(data,name,'exec')
            archive.writestr(name,data)
    payload=stream.getvalue();digest=hashlib.sha256(payload).hexdigest()
    code=f'''import base64,hashlib,io,zipfile,subprocess,sys
from pathlib import Path
payload=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={digest!r}
root=Path('/kaggle/working/cellpose_prepare_package');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:archive.extractall(root)
subprocess.run([sys.executable,'-u',str(root/'scripts/cellpose_prepare_runner.py'),str(root)],check=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=[dict(cell_type='markdown',metadata={},source=['# E028 Cellpose-DINO public preparation\nCPU only. Download public model and dependencies, then strict load and synthetic forward check. No competition data, training or submission.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))])
    meta=dict(id='jarturo/biohub-lab-cellpose-preparation-cpu',title='Biohub Lab Cellpose Preparation CPU',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=True,competition_sources=[],dataset_sources=[],kernel_sources=[],model_sources=[])
    folder=ROOT/'kaggle/cellpose_prepare';folder.mkdir(exist_ok=True)
    for name,value in [('notebook.ipynb',nb),('kernel-metadata.json',meta),('payload.json',dict(sha256=digest,files=files,contains_credentials=False,contains_images=False,contains_weights=False))]:(folder/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    receipt=dict(payload_sha256=digest,notebook_sha256=hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest(),files=len(files),gpu=False,private=True)
    (ROOT/'results/E028_PREP_preflight.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
