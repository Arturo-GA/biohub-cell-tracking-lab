import hashlib,json,sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate();root=Path('artifacts/research_latest_20260920');root.mkdir(exist_ok=True);records=[]
refs=['andnyu/biohub-density-adaptive-0-948-reproduction','ghazarosghazaros/biohub-dae-self-distill-repeat','noisyislands/biohub-linker-association-mlp-v2','anvithpothula/biohub-0-95','evgendvorkin/biohub-0-942-lb-proxy-score-0-9417','codezzzsleep/biohub-095-owned-validation']
refs=sys.argv[1:] or refs
for ref in refs:
    folder=root/ref.replace('/','__');folder.mkdir(exist_ok=True)
    try:
        api.kernels_pull(ref,str(folder),metadata=True,quiet=True)
        paths=list(folder.glob('*.ipynb'));path=paths[0] if paths else next(folder.glob('*.py'))
        if path.suffix=='.ipynb':
            nb=json.loads(path.read_text(encoding='utf8'));code='\n\n'.join(''.join(c.get('source',[])) for c in nb['cells'] if c['cell_type']=='code');md='\n\n'.join(''.join(c.get('source',[])) for c in nb['cells'] if c['cell_type']=='markdown')
        else:code=path.read_text(encoding='utf8');md=''
        (folder/'source.py').write_text(code,encoding='utf8');(folder/'description.md').write_text(md,encoding='utf8')
        record=dict(ref=ref,file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),code_lines=len(code.splitlines()),metadata=json.loads((folder/'kernel-metadata.json').read_text()))
        records.append(record);print(ref,len(code.splitlines()),flush=True)
    except Exception as e:records.append(dict(ref=ref,error=str(e)));print(ref,'ERROR',str(e),flush=True)
previous=json.loads((root/'downloads.json').read_text()) if (root/'downloads.json').exists() else []
records=[r for r in previous if r['ref'] not in refs]+records
(root/'downloads.json').write_text(json.dumps(records,indent=2)+'\n');Path('results/public_notebooks_latest_downloads.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='metadata'} for r in records],indent=2)+'\n')
