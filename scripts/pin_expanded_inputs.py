"""Verify completed E040 data counts and freeze its manifest before training."""
import json
from pathlib import Path
from build_three_lines import build
ROOT=Path(__file__).resolve().parents[1]
def main():
    receipt=json.loads((ROOT/'results/E040_PREPARE_completed.json').read_text());result=receipt['result'];cfg=json.loads((ROOT/'baseline/e040_protocol.json').read_text());assert result['config']==cfg
    counts={split:sum(r['positive'] for r in result['videos'] if r['split']==split) for split in ['fit','development']}
    assert counts==dict(fit=124,development=15),counts
    assert not set(cfg['fit'])&set(cfg['development'])
    pin=dict(manifest_sha256=receipt['manifest_sha256'],positive_counts=counts,source_kernel=receipt['kernel'],source_version=receipt['version'])
    (ROOT/'baseline/e040_input_pin.json').write_text(json.dumps(pin,indent=2)+'\n')
    for stage,runner,kernels,gpu in [('E040_TRAIN','scripts/expanded_events_train_runner.py',['jarturo/biohub-e040-prepare-cpu'],True),('E040_EVALUATE','scripts/expanded_events_evaluate_runner.py',['jarturo/biohub-e040-prepare-cpu','jarturo/biohub-e040-train'],False)]:
        files=json.loads((ROOT/'kaggle'/stage.lower()/'payload.json').read_text())['files']+['baseline/e040_input_pin.json'];build(stage,runner,files,kernels,gpu=gpu)
    print(json.dumps(pin))
if __name__=='__main__':main()
