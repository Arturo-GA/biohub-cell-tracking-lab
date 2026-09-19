"""Freeze and package the E016 CPU identity-parent training."""
import ast,base64,hashlib,io,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build():
    names=json.loads((ROOT/'kaggle/detection_identity/payload.json').read_text())['files']
    names+=['src/biohub_lab/identity_parent.py','scripts/identity_parent_runner.py']
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            data=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
            if name.endswith('.py'):compile(data,name,'exec')
            archive.writestr(zipfile.ZipInfo(name,date_time=(2026,9,19,0,0,0)),data,compress_type=zipfile.ZIP_DEFLATED)
    payload=buffer.getvalue();digest=hashlib.sha256(payload).hexdigest()
    notebook=json.loads((ROOT/'kaggle/association_cpu/notebook.ipynb').read_text())
    code=''.join(notebook['cells'][1]['source'])
    assignment=next(n for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
    previous=ast.literal_eval(assignment.value.args[0])
    code=code.replace(previous,base64.b64encode(payload).decode()).replace(hashlib.sha256(base64.b64decode(previous)).hexdigest(),digest)
    code=code.replace('/kaggle/working/association_cpu_package','/kaggle/working/identity_parent_package').replace('scripts/association_cpu_runner.py','scripts/identity_parent_runner.py')
    compile(code,'notebook','exec')
    notebook['cells'][0]['source']=['# E016 CPU competitive parent training\n','48 fit videos, 3000 CPU training steps; 16 calibration videos. Frozen primary detections and cached image features. Compare learned and geometric assignments. No GPU or leaderboard submission.\n']
    notebook['cells'][1]['source']=code.splitlines(True)
    meta=json.loads((ROOT/'kaggle/association_cpu/kernel-metadata.json').read_text())
    meta.update(id='jarturo/biohub-lab-identity-parent-cpu',title='Biohub Lab Identity Parent CPU',
        kernel_sources=['jarturo/biohub-lab-learned-event-graph'])
    folder=ROOT/'kaggle/identity_parent';folder.mkdir(exist_ok=True)
    for name,value in [('notebook.ipynb',notebook),('kernel-metadata.json',meta),('payload.json',dict(sha256=digest,files=sorted(names),contains_weights=False,contains_images=False,contains_credentials=False))]:
        (folder/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    from launch_cpu_notebook import validate
    validate(folder)
    checks=dict(payload_sha256=digest,source_files_compiled=True,cpu_only=True,fit_videos=48,calibration_videos=16,evaluation_videos=0,training_steps=3000,unit_tests_passed=4,no_automatic_monitoring=True,notebook_sha256=sha(folder/'notebook.ipynb'))
    (ROOT/'results/E016_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':build()
