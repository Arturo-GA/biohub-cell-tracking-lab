import base64,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'artifacts/research_20260920/hengck_detector_source/model_v5.py'
    pin=dict(model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),dataset='hengck23/hengck23-cell-point-detector-demo',weight='00000030.pth',license='unknown per dataset metadata; attached original for private diagnostic only',training_membership='unknown')
    (ROOT/'baseline/e031_source_pin.json').write_text(json.dumps(pin,indent=2)+'\n')
    files=['scripts/dense_detector_gpu_runner.py','baseline/e031_source_pin.json','baseline/e031_dense_detector.json'];stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            data=(ROOT/name).read_bytes()
            if name.endswith('.py'):compile(data,name,'exec')
            archive.writestr(name,data)
    payload=stream.getvalue();digest=hashlib.sha256(payload).hexdigest();folder=ROOT/'kaggle/dense_detector_gpu';folder.mkdir(exist_ok=True)
    code=f'''import base64,hashlib,io,zipfile,subprocess,sys,os
from pathlib import Path
payload=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={digest!r}
root=Path('/kaggle/working/dense_detector_gpu_package');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:archive.extractall(root)
subprocess.run([sys.executable,'-u',str(root/'scripts/dense_detector_gpu_runner.py'),str(root)],check=True,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
'''
    compile(code,'notebook','exec')
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=[dict(cell_type='markdown',metadata={},source=['# E031 dense detector\nNeural inference on all frames of 16 calibration videos; two normalization semantics. No training or leaderboard submission.']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))])
    meta=dict(id='jarturo/biohub-dense-detector-gpu',title='Biohub Dense Detector GPU',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,machine_shape='NvidiaTeslaT4',competition_sources=[],dataset_sources=[pin['dataset']],kernel_sources=['jarturo/biohub-dense-detector-prepare-cpu'],model_sources=[])
    for name,obj in [('notebook.ipynb',nb),('kernel-metadata.json',meta),('payload.json',dict(sha256=digest,files=files,contains_credentials=False,contains_weights=False,contains_images=False))]:(folder/name).write_text(json.dumps(obj,indent=2)+'\n')
    checks=dict(notebook_sha256=hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest(),payload_sha256=digest,gpu_only_neural_inference=True)
    (ROOT/'results/E031_GPU_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks)
if __name__=='__main__':main()
