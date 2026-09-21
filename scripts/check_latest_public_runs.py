import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate();root=Path('artifacts/research_latest_20260920');records=[]
for ref in ['andnyu/biohub-density-adaptive-0-948-reproduction','ghazarosghazaros/biohub-dae-self-distill-repeat','noisyislands/biohub-linker-association-mlp-v2']:
    folder=root/ref.replace('/','__')/'outputs';folder.mkdir(exist_ok=True)
    try:
        status=api.kernels_status(ref).status.name;files,_=api.kernels_output(ref,str(folder),file_pattern=r'.*(metrics.json|prov.json)$',quiet=True)
        records.append(dict(ref=ref,status=status,files=files));print(ref,status,flush=True)
    except Exception as e:records.append(dict(ref=ref,error=str(e)));print(ref,str(e),flush=True)
Path('results/public_notebooks_latest_run_status.json').write_text(json.dumps(records,indent=2)+'\n')
