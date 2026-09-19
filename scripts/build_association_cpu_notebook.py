"""Package E014 decoding/evaluation as a private CPU notebook."""
import ast, base64, hashlib, io, json, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build():
    completed=json.loads((ROOT/'results/E014_completed.json').read_text())
    pins=dict(e014_result_sha256=completed['result_sha256'],
        frozen_inputs_sha256='1c8f7d3d1134db7ed3d80355a3a61eac8b99bf2dd9dec68fc50362d16b6fb5c0',
        checkpoint_sha256={k:v['checkpoint_sha256'] for k,v in completed['heads'].items()},
        thresholds=completed['thresholds'],paired_control_score=.9092570108739322,
        arms=['continuations','divisions'],selection='Highest official score on the 16 calibration videos; ties favor continuations',
        quality='Neutral constant 0.5',solver='Existing overlapping MILP with recorded fallback; default 5 second window time limit')
    (ROOT/'baseline/e014_cpu_pins.json').write_text(json.dumps(pins,indent=2)+'\n')
    names=['src/biohub_lab/'+n+'.py' for n in ('__init__','association_rank','association_cpu','event_data',
           'event_solver','detector_proposals','evaluate','submission')]
    names += ['src/biohub_official/'+n for n in ('__init__.py','metrics.py','division_metrics.py','LICENSE')]
    names += ['scripts/association_cpu_runner.py','baseline/e014_cpu_pins.json','baseline/event_graph_split.json','NOTICE.md']
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            data=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
            if name.endswith('.py'):compile(data,name,'exec')
            archive.writestr(zipfile.ZipInfo(name,date_time=(2026,9,19,0,0,0)),data,compress_type=zipfile.ZIP_DEFLATED)
    payload=buffer.getvalue();digest=hashlib.sha256(payload).hexdigest()
    # Reuse the already validated offline dependency installation, without changing that notebook.
    control=json.loads((ROOT/'kaggle/cpu_control/notebook.ipynb').read_text())
    code=''.join(control['cells'][1]['source'])
    old_tree=ast.parse(code)
    old_payload=next(n for n in old_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
    old_encoded=ast.literal_eval(old_payload.value.args[0]);old_digest=hashlib.sha256(base64.b64decode(old_encoded)).hexdigest()
    code=code.replace(old_encoded,base64.b64encode(payload).decode()).replace(old_digest,digest)
    code=code.replace('/kaggle/working/cpu_package','/kaggle/working/association_cpu_package')
    code=code.replace('scripts/kaggle_cpu_control.py','scripts/association_cpu_runner.py');ast.parse(code)
    control['cells'][0]['source']=['# Biohub E014 CPU association evaluation\n',
        'Reuses trained E014 checkpoints and E013 image features. CPU only; no training or image inference.\n',
        'Compare two decoding arms on 16 calibration videos; freeze selection before evaluating 48 development videos.\n',
        'Official metric, fixed hashes, saved per-video predictions and explicit solver fallback reports. No leaderboard submission.\n']
    control['cells'][1]['source']=code.splitlines(True)
    metadata=dict(id='jarturo/biohub-lab-association-cpu-evaluation',title='Biohub Lab Association CPU Evaluation',
        code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,
        enable_gpu=False,enable_tpu=False,enable_internet=False,
        competition_sources=['biohub-cell-tracking-during-development'],
        dataset_sources=['pilkwang/biohub-tracking-support-pack-50ep-v1'],
        kernel_sources=['jarturo/biohub-lab-learned-event-graph','jarturo/biohub-lab-association-ranking'],model_sources=[])
    folder=ROOT/'kaggle/association_cpu';folder.mkdir(exist_ok=True)
    for name,value in [('notebook.ipynb',control),('kernel-metadata.json',metadata),
        ('payload.json',dict(sha256=digest,files=sorted(names),contains_weights=False,contains_images=False,contains_credentials=False))]:
        (folder/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    from launch_cpu_notebook import validate
    validate(folder)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert archive.namelist()==sorted(names)
    checks=dict(payload_sha256=digest,source_files_verified=len(names),cpu_only=True,
        unit_tests_passed=3,calibration_videos=16,evaluation_videos=48,training_started=False,
        no_automatic_monitoring=True,notebook_sha256=hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest())
    (ROOT/'results/E014_CPU_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))


if __name__=='__main__':build()
