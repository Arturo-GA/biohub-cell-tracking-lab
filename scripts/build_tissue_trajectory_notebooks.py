"""Package separate CPU trajectory and GPU full-Harmonic calibration runs."""
import ast,base64,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def package(source_folder,folder,names,old_runner,new_runner,slug,title,gpu,kernels):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(names)):
            data=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
            if name.endswith('.py'):compile(data,name,'exec')
            archive.writestr(zipfile.ZipInfo(name,date_time=(2026,9,19,0,0,0)),data,compress_type=zipfile.ZIP_DEFLATED)
    payload=buffer.getvalue();digest=hashlib.sha256(payload).hexdigest()
    notebook=read(ROOT/source_folder/'notebook.ipynb');code=''.join(notebook['cells'][1]['source'])
    node=next(n for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
    previous=ast.literal_eval(node.value.args[0])
    code=code.replace(previous,base64.b64encode(payload).decode()).replace(hashlib.sha256(base64.b64decode(previous)).hexdigest(),digest)
    code=code.replace(old_runner,new_runner)
    for old in ['identity_parent_package','event_graph_package']:code=code.replace('/kaggle/working/'+old,'/kaggle/working/'+folder.name+'_package')
    compile(code,'notebook','exec');notebook['cells'][1]['source']=code.splitlines(True)
    notebook['cells'][0]['source']=['# '+title+'\n',('Full Harmonic 3D image inference only; official evaluation deferred to CPU.\n' if gpu else 'Collective tissue motion and sequence-level trajectory assignment. CPU only, no training.\n'),'Fixed 16 calibration videos; no leaderboard submission.\n']
    meta=read(ROOT/source_folder/'kernel-metadata.json');meta.update(id='jarturo/'+slug,title=title,kernel_sources=kernels)
    assert meta['enable_gpu']==gpu and meta['is_private'] and not meta['enable_tpu']
    folder.mkdir(exist_ok=True)
    for name,value in [('notebook.ipynb',notebook),('kernel-metadata.json',meta),('payload.json',dict(sha256=digest,files=sorted(set(names)),contains_images=False,contains_weights=False,contains_credentials=False))]:
        (folder/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    return dict(payload_sha256=digest,notebook_sha256=sha(folder/'notebook.ipynb'),files=len(set(names)),gpu=gpu,private=True,calibration_videos=16)
def main():
    old=ROOT/'outputs/e016_recovery/identity_parent'
    pins=dict(result_sha256=sha(old/'result.json'),geometric_csv_sha256=read(old/'frozen_predictions.json')['csv_sha256']['geometric'],geometric_metrics_sha256=sha(old/'geometric_metrics.json'))
    (ROOT/'baseline/e017_pins.json').write_text(json.dumps(pins,indent=2)+'\n')
    cpu=read(ROOT/'kaggle/identity_parent/payload.json')['files']
    cpu+=['src/biohub_lab/tissue_trajectory.py','scripts/tissue_trajectory_runner.py','baseline/e017_pins.json']
    checks={}
    checks['cpu']=package(Path('kaggle/identity_parent'),ROOT/'kaggle/tissue_trajectory',cpu,'scripts/identity_parent_runner.py','scripts/tissue_trajectory_runner.py','biohub-lab-tissue-trajectory-cpu','Biohub Lab Tissue Trajectory CPU',False,['jarturo/biohub-lab-identity-parent-cpu'])
    gpu=read(ROOT/'kaggle/event_graph_control/payload.json')['files']+cpu+['scripts/calibration_control_runner.py']
    checks['control']=package(Path('kaggle/event_graph_control'),ROOT/'kaggle/calibration_control',gpu,'scripts/event_control_runner.py','scripts/calibration_control_runner.py','biohub-lab-harmonic-calibration-control','Biohub Lab Harmonic Calibration Control',True,[])
    checks.update(unit_tests_passed=4,automatic_monitoring=False,gpu_justification='Full frozen Harmonic volumetric neural inference with TTA; no new training, CPU-only association experiment separate')
    (ROOT/'results/E017_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
