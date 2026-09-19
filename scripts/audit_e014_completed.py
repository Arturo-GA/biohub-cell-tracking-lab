"""Verify recovered E014 artifacts and recompute known-candidate diagnostics."""
import ast, base64, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
from biohub_lab.association_rank import average_precision

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    folder=ROOT/'outputs/e014_recovery';root=folder/'association_rank_experiment'
    result=read(root/'result.json');training=read(root/'training.json');cal=read(root/'calibration.json')
    assert result['status']==training['status']=='complete' and training['history'][-1]['step']==6000
    assert sha(root/'input_identity.json')==result['input_identity_sha256']
    identity=read(root/'input_identity.json');split=read(ROOT/'baseline/event_graph_split.json')['split']
    assert identity['split']==split
    assert set(identity['videos'])==set(split['fit']+split['calibration'])
    nb=read(ROOT/'kaggle/association_rank/notebook.ipynb');tree=ast.parse(''.join(nb['cells'][1]['source']))
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
    payload=base64.b64decode(ast.literal_eval(node.value.args[0]))
    assert hashlib.sha256(payload).hexdigest()==read(ROOT/'results/E014_launch.json')['payload_sha256']
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():assert archive.read(name)==(folder/'association_rank_package'/name).read_bytes()
        source_count=len(archive.namelist())
    pooled={};thresholds={}
    from biohub_lab.association_cpu import calibrated_threshold
    for head in ('edge','division'):
        assert sha(root/f'best_{head}.pt')==result['heads'][head]['checkpoint_sha256']
        scores=[];labels=[];skills=[]
        for name in split['calibration']:
            path=root/'calibration'/name/f'{head}_known_scores.npz'
            if not path.exists():
                assert name not in cal[head]['per_video'];continue
            with np.load(path,allow_pickle=False) as d:s,y=d['scores'],d['labels']
            ap=average_precision(s,y);record=cal[head]['per_video'][name]
            if ap is not None:
                assert np.isclose(ap,record['average_precision'],rtol=0,atol=1e-12)
                skills.append((ap-y.mean())/(1-y.mean()))
            scores.append(s);labels.append(y)
        assert np.isclose(np.mean(skills),result['heads'][head]['selection_skill'],atol=1e-6,rtol=0)
        scores=np.concatenate(scores);labels=np.concatenate(labels)
        pooled[head]=dict(average_precision=average_precision(scores,labels),positives=int(labels.sum()),examples=len(labels))
        thresholds[head]=calibrated_threshold(scores,labels,1. if head=='edge' else .5)
    receipt=dict(status='complete',steps=6000,source_files_verified=source_count,
        result_sha256=sha(root/'result.json'),input_identity_sha256=sha(root/'input_identity.json'),
        heads=result['heads'],pooled_calibration=pooled,thresholds=thresholds,
        training_seconds=training['history'][-1]['seconds'],total_process_seconds=result['seconds'],
        official_score_available=False,leaderboard_submitted=False,
        scope='Known candidate calibration; full trajectory evaluation pending')
    (ROOT/'results/E014_completed.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
