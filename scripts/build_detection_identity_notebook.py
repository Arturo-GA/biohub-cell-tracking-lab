"""Freeze and package the E015 CPU identity ablation."""
import ast,base64,hashlib,io,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build():
    old=ROOT/'outputs/e014_cpu_recovery/association_cpu'
    pins=dict(e014_cpu_result_sha256=sha(old/'result.json'),
        e013_frozen_sha256='1c8f7d3d1134db7ed3d80355a3a61eac8b99bf2dd9dec68fc50362d16b6fb5c0',
        calibration_frozen_sha256=sha(old/'calibration_predictions_frozen.json'),
        baseline_csv_sha256=json.loads((old/'calibration_predictions_frozen.json').read_text())['csv_sha256']['continuations'],
        baseline_metrics_sha256=sha(old/'calibration_continuations_metrics.json'))
    (ROOT/'baseline/e015_identity_pins.json').write_text(json.dumps(pins,indent=2)+'\n')
    names=json.loads((ROOT/'kaggle/association_cpu/payload.json').read_text())['files']
    names+=['src/biohub_lab/detection_identity.py','src/biohub_lab/temporal_data.py',
            'scripts/detection_identity_runner.py','baseline/e015_identity_pins.json']
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
    code=code.replace('/kaggle/working/association_cpu_package','/kaggle/working/detection_identity_package').replace('scripts/association_cpu_runner.py','scripts/detection_identity_runner.py')
    compile(code,'notebook','exec')
    notebook['cells'][0]['source']=['# E015 CPU detection identity audit\n','Cached calibration graphs only. Freeze annotation-free projection before diagnosis and official evaluation. No training, GPU or leaderboard submission.\n']
    notebook['cells'][1]['source']=code.splitlines(True)
    meta=json.loads((ROOT/'kaggle/association_cpu/kernel-metadata.json').read_text())
    meta.update(id='jarturo/biohub-lab-detection-identity-audit',title='Biohub Lab Detection Identity Audit',
        kernel_sources=['jarturo/biohub-lab-learned-event-graph','jarturo/biohub-lab-association-cpu-evaluation'])
    folder=ROOT/'kaggle/detection_identity';folder.mkdir(exist_ok=True)
    for name,value in [('notebook.ipynb',notebook),('kernel-metadata.json',meta),('payload.json',dict(sha256=digest,files=sorted(names),contains_weights=False,contains_images=False,contains_credentials=False))]:
        (folder/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    from launch_cpu_notebook import validate
    validate(folder)
    checks=dict(payload_sha256=digest,source_files_compiled=True,cpu_only=True,calibration_videos=16,evaluation_videos=0,training_started=False,no_automatic_monitoring=True,notebook_sha256=sha(folder/'notebook.ipynb'))
    (ROOT/'results/E015_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':build()
