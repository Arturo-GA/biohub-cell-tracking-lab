import base64,hashlib,io,json,zipfile
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,package,sha

def main():
    r=json.loads((ROOT/'outputs/e033_eval/temporal_volume_evaluation/result.json').read_text());assert r['proceed_to_full_graph']
    config=dict(experiment='E033-graph',original_edge_prior=1.,image_manifest_sha256=json.loads((ROOT/'results/E031_PREP_completed.json').read_text())['result_sha256'],training_manifest_sha256=json.loads((ROOT/'results/E033_TRAIN_completed.json').read_text())['manifest_sha256'],control_sha256=json.loads((ROOT/'baseline/e018_pins.json').read_text())['harmonic_csv_sha256'],control_score=json.loads((ROOT/'results/E031_completed.json').read_text())['control']['score'],protocol='Fixed complete continuation assignment, original-edge prior1,12 nearest parents within20um plus original, cosine temperature0.2. All nodes and degrees preserved; divisions and gap edges locked. No parameter sweep.')
    (ROOT/'baseline/e033_graph.json').write_text(json.dumps(config,indent=2)+'\n')
    for stage in ['prepare','features','evaluate']:
        if (ROOT/('results/E033_GRAPH_'+stage.upper()+'_launch.json')).exists():continue
        folder=ROOT/('kaggle/temporal_graph_'+stage);runner='scripts/temporal_graph_'+stage+'_runner.py'
        files=['baseline/e033_graph.json',runner,'src/biohub_lab/temporal_volume.py','src/biohub_lab/temporal_graph.py']
        if stage!='features':
            files+=json.loads((ROOT/'kaggle/identity_parent/payload.json').read_text())['files']
            kernels=['jarturo/biohub-dense-detector-prepare-cpu','jarturo/biohub-lab-full-calibration-compare-cpu'] if stage=='prepare' else ['jarturo/biohub-temporal-graph-prepare-cpu','jarturo/biohub-temporal-graph-features']
            checks=package(Path('kaggle/identity_parent'),folder,files,'scripts/identity_parent_runner.py',runner,'biohub-temporal-graph-'+stage+'-cpu','Biohub Temporal Graph '+stage.title()+' CPU',False,kernels)
            nb=json.loads((folder/'notebook.ipynb').read_text());nb['cells'][0]['source']=['# E033 full graph '+stage+'\nEncoder passed ranking gate. Preserve Harmonic nodes, divisions and degrees; paired geometry and appearance comparison.']
            (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n');checks['notebook_sha256']=sha(folder/'notebook.ipynb')
        else:
            stream=io.BytesIO()
            with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
                for name in files:
                    content=(ROOT/name).read_text(encoding='utf8')
                    if name.endswith('.py'):compile(content,name,'exec')
                    archive.writestr(name,content)
            payload=stream.getvalue();digest=hashlib.sha256(payload).hexdigest();folder.mkdir(exist_ok=True)
            code=f'''import base64,hashlib,io,zipfile,subprocess,sys,os
from pathlib import Path
payload=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={digest!r}
root=Path('/kaggle/working/temporal_graph_features_package');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:archive.extractall(root)
subprocess.run([sys.executable,'-u',str(root/{runner!r}),str(root)],env=dict(os.environ,PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='2'),check=True)
'''
            nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python')),cells=[dict(cell_type='markdown',metadata={},source=['# E033 full Harmonic node embeddings\nGPU encoder inference only; CPU normalization and graph evaluation separate.']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))])
            meta=dict(id='jarturo/biohub-temporal-graph-features',title='Biohub Temporal Graph Features',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,machine_shape='NvidiaTeslaT4',competition_sources=[],dataset_sources=[],kernel_sources=['jarturo/biohub-temporal-graph-prepare-cpu','jarturo/biohub-temporal-volume-train'],model_sources=[])
            for name,value in [('notebook.ipynb',nb),('kernel-metadata.json',meta),('payload.json',dict(sha256=digest,files=files,contains_credentials=False,contains_weights=False,contains_images=False))]:(folder/name).write_text(json.dumps(value,indent=2)+'\n')
            checks=dict(notebook_sha256=sha(folder/'notebook.ipynb'),payload_sha256=digest,gpu=True,private=True)
        (ROOT/('results/E033_GRAPH_'+stage.upper()+'_preflight.json')).write_text(json.dumps(checks,indent=2)+'\n');print(stage,checks)
if __name__=='__main__':main()
