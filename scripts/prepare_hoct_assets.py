"""Package PUBLIC HOCT weights only; diagnostic data remains on Kaggle."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / 'artifacts/hoct_public_weights'
folder.mkdir(parents=True, exist_ok=True)
expected = '5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33'
shutil.copyfile(ROOT / 'artifacts/hoct_assets/general_v1.pt', folder / 'general_v1.pt')
assert hashlib.sha256((folder / 'general_v1.pt').read_bytes()).hexdigest() == expected
shutil.copyfile(ROOT / 'licenses/HOCT.txt', folder / 'HOCT_LICENSE.txt')
receipt = json.loads((ROOT / 'outputs/diagnostic/run_receipt.json').read_text())
split = json.loads((ROOT / 'outputs/diagnostic/biohub_control/secondary_seed_weights/unet_transformer/split_0/split_manifest.json').read_text())
overlap = sorted(set(receipt['datasets']) & set(split['train']))
assert len(overlap) == len(receipt['datasets'])
receipt['scope'] = 'IN-SAMPLE: every diagnostic video belongs to secondary checkpoint training set'
receipt['checkpoint_train_overlap'] = overlap
receipt['interpretation'] = 'No additional annotated correct edges/divisions; E001 closed without demonstrated gain.'
(ROOT / 'results/E001_completed.json').write_text(json.dumps(receipt, indent=2) + '\n')
(folder / 'hoct_model_manifest.json').write_text(json.dumps({
    'hoct_source': 'https://github.com/royerlab/hoct',
    'hoct_commit': '2ccc5040823bc944ab67790abd1f56eea7cd4f05',
    'model_url': 'https://github.com/royerlab/hoct/releases/download/weights-v1/general_v1.pt',
    'model_sha256': expected,
}, indent=2) + '\n')
(folder / 'dataset-metadata.json').write_text(json.dumps({
    'title': 'Biohub Lab HOCT Public Weights',
    'id': 'jarturo/biohub-lab-hoct-public-weights',
    'licenses': [{'name': 'other'}],
    'description': 'Official public HOCT general_v1 weights, SHA256 pinned, MIT license included. '
        'Only public model weights and their public provenance/license are included. No user or competition data.',
}, indent=2) + '\n')
assert {p.name for p in folder.iterdir()} == {'general_v1.pt','HOCT_LICENSE.txt','hoct_model_manifest.json','dataset-metadata.json'}
print(folder)
