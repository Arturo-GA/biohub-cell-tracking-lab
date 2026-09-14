"""Package pinned PUBLIC CELLECT weights with license and provenance only."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
COMMIT='3586070926f7f1fd5d8df37456861d22bdc63236'
WEIGHT='U-ext+-x3rd-149.0-4.6540.pth'
SHA='f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5'


def main():
    source=ROOT/'third_party/cellect/model'/WEIGHT
    assert hashlib.sha256(source.read_bytes()).hexdigest()==SHA
    folder=ROOT/'artifacts/cellect_public_weights'
    folder.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,folder/WEIGHT)
    shutil.copyfile(ROOT/'licenses/CELLECT.txt',folder/'CELLECT_LICENSE.txt')
    (folder/'cellect_manifest.json').write_text(json.dumps(dict(
        repository='https://github.com/zzz333za/CELLECT', commit=COMMIT,
        url=f'https://raw.githubusercontent.com/zzz333za/CELLECT/{COMMIT}/model/{WEIGHT}',
        weight_sha256=SHA, source_license='GPL-2.0; repository license included',
        contents='Public checkpoint and public provenance/license only; no user/competition data'),indent=2)+'\n')
    (folder/'dataset-metadata.json').write_text(json.dumps(dict(
        id='jarturo/biohub-lab-cellect-public-weights',title='Biohub Lab CELLECT Public Weights',
        licenses=[{'name':'other'}], description='Pinned official public CELLECT detector checkpoint. '
        'GPL-2.0 repository license and public provenance included. No competition data or user predictions.'),indent=2)+'\n')
    assert {p.name for p in folder.iterdir()}=={WEIGHT,'CELLECT_LICENSE.txt','cellect_manifest.json','dataset-metadata.json'}
    print(folder)


if __name__=='__main__': main()
