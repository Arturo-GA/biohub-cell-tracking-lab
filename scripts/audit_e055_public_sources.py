"""Static source comparison only; downloaded notebook code is not executed."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path('artifacts/research_latest_20260920')

def tree(slug):
    path = ROOT / slug / 'source.py'
    return path, ast.parse(path.read_text(encoding='utf8'))

def executable(node):
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)

def main():
    base_path, base = tree('raunakdey07__biohub-harmonic-fusion-v3')
    records = []
    for slug in ['beraterolelk__0-947-lb-biohub-deepcenter-ilp-tracker', 'mthem77__biohub-b0-harmonic-fusion-safety-re-run', 'andnyu__biohub-947-mutual-rank', 'andnyu__biohub-947-synthetic-edge', 'haideptry__biohub-0-951-sota-deepcenter-fast-ilp-19m']:
        path, parsed = tree(slug)
        records.append(dict(ref=slug.replace('__','/'), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), same_executable_ast_as_v3=executable(parsed)==executable(base)))
    _, old = tree('andnyu__biohub-density-adaptive-0-948-reproduction')
    _, new = tree('haideptry__biohub-0-951-sota-deepcenter-fast-ilp-19m')
    classes = ['_DivNetConvBlock', 'DivNetMitosisClassifier']
    same = {name: [ast.dump(n) for n in ast.walk(old) if isinstance(n,ast.ClassDef) and n.name==name] == [ast.dump(n) for n in ast.walk(new) if isinstance(n,ast.ClassDef) and n.name==name] for name in classes}
    result = dict(base_source_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(), sources=records,
        divnet_classes_unchanged_from_previously_audited_incompatible_model=same,
        divnet_limit='New source still uses strict=False. For the previously downloaded checkpoint, identical classes load zero tensors. This is not proof that every alternative checkpoint is incompatible.',
        mutual_rank='New hook adds rank bonus beta=.12 to logits before softmax/ILP; E051 was only a postprocess adaptation, not an exact reproduction.',
        synthetic_edge='New third UNetNodeTransformer: column margin <.12, synthetic margin advantage >.02, weight <=.25 varying by uncertainty; checkpoint SHA 0eacacaf0b43bfd5a063495d6991a4911cd045c37650363d5d8826a0e7ed3dd9. Not yet locally validated.',
        score_evidence=[dict(url='https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/728324',level='staff announcement',finding='Metric patched, submissions rescored; staff says complete.'),dict(url='https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/736937',level='participant report',finding='Some public notebook list scores remain from before rescore.')],
        score_limit='No evidence yet that the specific V3 0.947 label is stale. Visible CSV equality does not establish hidden-test equality or metric-version equality.')
    Path('results/E055_public_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
