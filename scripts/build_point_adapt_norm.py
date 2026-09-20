"""Package E025 normalization diagnosis on CPU, with frozen original/adapted weights."""
import base64,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    files=['scripts/point_adapt_norm_diagnostic.py','baseline/e025_point_adapt.json','vendor/nucverse/attention_unet_3d.py','vendor/nucverse/LICENSE','NOTICE.md']
    stream=io.BytesIO()
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
root=Path('/kaggle/working/point_norm_package');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:archive.extractall(root)
subprocess.run([sys.executable,'-u',str(root/'scripts/point_adapt_norm_diagnostic.py'),str(root)],check=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=[dict(cell_type='markdown',metadata={},source=['# E025 normalization diagnostic\nCPU only. Paired deterministic forward modes; eight fixed crops, no training or submission.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))])
    meta=dict(id='jarturo/biohub-lab-point-normalization-cpu',title='Biohub Lab Point Normalization CPU',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,competition_sources=[],dataset_sources=[],kernel_sources=['jarturo/biohub-lab-nucverse-preparation-cpu','jarturo/biohub-lab-point-adapt-data-cpu','jarturo/biohub-lab-point-adapt-training'],model_sources=[])
    folder=ROOT/'kaggle/point_norm';folder.mkdir(exist_ok=True)
    for name,value in [('notebook.ipynb',nb),('kernel-metadata.json',meta),('payload.json',dict(sha256=digest,files=files,contains_credentials=False,contains_images=False,contains_weights=False))]:(folder/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    result=dict(payload_sha256=digest,notebook_sha256=hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest(),files=len(files),gpu=False,private=True)
    (ROOT/'results/E025_NORM_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
