"""Require the registered notebook to reproduce its locally checked graph."""
import json
from pathlib import Path
import sys
from finalize_e071 import read_and_validate,graph_signature,ROOT

name=sys.argv[1]
shapes=json.loads((ROOT/'results/E054_CONTROL_completed.json').read_text())['result']['shapes']
local=json.loads((ROOT/'outputs/e071/local'/name/'verified.json').read_text())
remote=read_and_validate(ROOT/'outputs/e071'/name/'submission.csv',shapes)
result={s:graph_signature(*g)[3]==local['graph_signatures'][s] for s,g in remote.items()}
print(json.dumps(result))
assert all(result.values()),'Remote graph differs from local preparation; inspect before launching more candidates'
