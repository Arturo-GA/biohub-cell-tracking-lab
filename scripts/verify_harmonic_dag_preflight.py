"""Exercise E012 using real public weights and a small synthetic image movie."""
import ast
import base64
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import zipfile

import numpy as np
import torch
import zarr
from biohub_lab.harmonic_centers import CHECKPOINTS,DETECTOR_CONFIG,detector_source,image_metadata
from harmonic_dag_runner import generate_video,sha

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'artifacts/e012_detector_check'


def main():
    torch.set_num_threads(2)
    baseline=(ROOT/'baseline/harmonic_inference.py').read_text()
    parsed=ast.parse(baseline)
    expected=next(ast.literal_eval(n.value) for n in parsed.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='_support_expected_sha256' for t in n.targets))
    for path in (ASSETS/'repo').rglob('*.py'):
        relative=path.relative_to(ASSETS/'repo').as_posix()
        if relative in expected:assert sha(path)==expected[relative],relative
    for name,digest in CHECKPOINTS.items():assert sha(ASSETS/name/'edge_predictor_best.pth')==digest
    # Materialize the same full runtime patch sequence over the pinned source.
    work=ASSETS/'smoke';work.mkdir(exist_ok=True);repo=work/'repo';(repo/'scripts').mkdir(exist_ok=True,parents=True)
    predictor=repo/'scripts/predict_unet_transformer.py'
    predictor.write_bytes((ASSETS/'repo/scripts/predict_unet_transformer.py').read_bytes())
    start='_ps = REPO_DIR / "scripts" / "predict_unet_transformer.py"'
    end='print("secondary edge-feature TTA patch installed and enabled", flush=True)'
    assert baseline.count(start)==baseline.count(end)==1
    block=start+baseline.split(start)[1].split(end)[0]+end
    env={'BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT':'.15','BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION':'.90'}
    with patch.dict(os.environ,env),redirect_stdout(io.StringIO()):
        exec(compile(block,'<actual-harmonic-runtime-patches>','exec'),{'REPO_DIR':repo,'WORKING_DIR':work,'os':os,'Path':Path})
    patched=predictor.read_text();trainer=(ASSETS/'repo/scripts/train_unet_transformer.py').read_text()
    source=detector_source(patched,trainer)
    # Verify retained window statements are bytecode-input equivalent as ASTs.
    before=next(n for n in ast.parse(patched).body if isinstance(n,ast.FunctionDef) and n.name=='predict_video')
    after=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='predict_video')
    old_loop=next(n for n in before.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='ws')
    new_loop=next(n for n in after.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='ws')
    retained=[n for n in old_loop.body if not (
        isinstance(n,ast.For) and ast.unparse(n.iter)=='range(W - 1)' or
        isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='coords_so_far' for t in n.targets))]
    assert [ast.dump(n) for n in retained]==[ast.dump(n) for n in new_loop.body]
    module_path=work/'image_detector.py'
    module_path.write_text(source.replace('/kaggle/working',work.as_posix()))
    sys.path.insert(0,str(ASSETS/'repo/src'))
    spec=importlib.util.spec_from_file_location('e012_smoke_detector',module_path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    primary,window,downsample=module.load_model(ASSETS/'primary/edge_predictor_best.pth',torch.device('cpu'))
    secondary,window2,downsample2=module.load_model(ASSETS/'secondary/edge_predictor_best.pth',torch.device('cpu'))
    assert window==window2==2 and downsample==downsample2==(1,4,4)
    cfg=module.PredictConfig(det_threshold=DETECTOR_CONFIG['det_threshold'],det_tta=True,pool_kernel_um=3.)
    rng=np.random.default_rng(12012);shape=(5,16,64,64)
    grid=np.stack(np.meshgrid(np.arange(16)*1.625,np.arange(64)*.40625,np.arange(64)*.40625,indexing='ij'),axis=-1)
    frames=[]
    for t in range(shape[0]):
        frame=30+rng.normal(0,2,shape[1:])
        for point in ([10,9,8+t*.4],[12,18,18-t*.3]):frame+=1000*np.exp(-np.sum((grid-point)**2,axis=-1)/4.)
        frames.append(np.rint(frame).clip(0,65535).astype(np.uint16))
    image=work/'synthetic.zarr';g=zarr.open_group(str(image),mode='w');g.create_array('0',data=np.stack(frames))
    g.attrs['image_statistics']={'quantiles':{'0.001':25.,'0.999':1000.}}
    image.with_suffix('.geff').write_text('must never be opened by inference')
    original_open=zarr.open_group;opened=[]
    def image_only(path,*args,**kwargs):
        if str(path).endswith('.geff'):raise AssertionError('Annotation access during image inference')
        opened.append(str(path));return original_open(path,*args,**kwargs)
    def forbidden(*args,**kwargs):raise AssertionError('Association function executed')
    primary.predict_edges=forbidden;secondary.predict_edges=forbidden
    with patch.dict(os.environ,dict(env,BIOHUB_EDGE_FEATURE_TTA='0',BIOHUB_SECONDARY_EDGE_FEATURE_TTA='0',BIOHUB_DIAGNOSTIC_ARM='')),patch.object(zarr,'open_group',side_effect=image_only):
        coords,edges=module.predict_video(primary,image,torch.device('cpu'),cfg,window_size=window,
            downsample=downsample,secondary_model=secondary,secondary_detection_weight=.80)
    assert not edges and len(coords)>0 and set(coords[:,0])==set(range(shape[0]))
    # Cached proposals here are synthetic; actual E011 files are checked separately below.
    cached=work/'cached';folder=cached/'videos/synthetic';folder.mkdir(parents=True,exist_ok=True)
    for method in ('cellect','gaussian'):
        np.savez_compressed(folder/(method+'.npz'),coords=np.array([[2,8,36,48]],np.float32),scores=np.ones(1),shape=shape)
    out=work/'result';result=generate_video('synthetic',coords,shape,out,cached,dict(checkpoint_sha256=CHECKPOINTS))
    assert result['primary_centers_preserved']==len(coords)
    pins=json.loads((ROOT/'baseline/harmonic_dag_inputs.json').read_text())
    cached_real=ROOT/'outputs/e011_v1/detection_dag_experiment'
    assert sha(cached_real/'frozen_inputs.json')==pins['frozen_sha256']
    for name,files in pins['files'].items():
        for relative,digest in files.items():assert sha(cached_real/'videos'/name/relative)==digest
    notebook=json.loads((ROOT/'kaggle/harmonic_dag/notebook.ipynb').read_text())
    code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    payload_node=next(n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
    payload=base64.b64decode(ast.literal_eval(payload_node.value.args[0]));digest=hashlib.sha256(payload).hexdigest()
    assert digest==json.loads((ROOT/'kaggle/harmonic_dag/payload.json').read_text())['sha256']
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():assert archive.read(name)==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    receipt=dict(experiment='E012',payload_sha256=digest,actual_public_checkpoints_verified=CHECKPOINTS,
        public_architecture_and_predictor_hashes_verified=True,full_runtime_patch_sequence_executed=True,
        retained_detector_window_statements_unchanged=True,actual_model_cpu_synthetic_inference=True,
        association_calls_blocked=True,annotation_access_blocked_during_inference=True,
        frames=shape[0],synthetic_primary_centers=len(coords),synthetic_combined_centers=result['arms']['combined']['nodes'],
        primary_centers_preserved=True,cached_E011_files_verified=240,all_packaged_files_verified=True,
        scope='Technical preflight on synthetic images with real public detector weights. No real annotation coverage or performance estimate.')
    (ROOT/'results/E012_detector_smoke.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
