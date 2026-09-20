"""Fetch pinned public files needed to reproduce verify_spatial_adapter.py."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    cfg=json.loads((ROOT/'baseline/e038_protocol.json').read_text());out=ROOT/'artifacts/e038_source';out.mkdir(parents=True,exist_ok=True)
    files=['LICENSE']+['src/spatialdino/models/layers/'+name+'.py' for name in ['attention','drop_path','layer_scale','mlp','block','patch_embed','pos_embed','encoder']]
    for path in files:
        url=f'https://raw.githubusercontent.com/kirchhausenlab/spatialdino/{cfg["source_commit"]}/{path}'
        urllib.request.urlretrieve(url,out/path.replace('/','__'))
    weight=out/'backbone.pth'
    if not weight.exists():urllib.request.urlretrieve('https://spatialdino.s3.amazonaws.com/models/spatial_dino/step=249999/backbone.pth',weight)
    with weight.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==cfg['weight_sha256']
    print('Pinned public reference ready; external files remain outside Git.')
if __name__=='__main__':main()
