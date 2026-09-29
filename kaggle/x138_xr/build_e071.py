"""Freeze final-round candidates as minimal edits of the confirmed c3 notebook."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / 'kaggle/x138_xr'
RES = ROOT / 'results/E071'
BASE = W / 'kernels/k_c3/notebook.ipynb'
BASE_SHA = '052f47600befa2afc0cac6892a1f14aa0381dc786b5c322b0dd51752070b598b'
CANDIDATES = {
    'C3_D04_R094': ('0.4', '0.94', 'Reported pair on c3; interaction is unproven on our base.'),
    'C3_R094': ('1.2', '0.94', 'Isolate localized readmission; retain original ILP cost.'),
    'C3_D04': ('0.4', '0.965', 'Isolate reduced ILP division cost; retain original readmission.'),
    'C3_D08_R094': ('0.8', '0.94', 'Intermediate ILP intervention with reported readmission threshold.'),
    'C3_D04_R092': ('0.4', '0.92', 'Exploratory extension of localized recovery, keeping the 4 um gate.'),
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def text(cell):
    return ''.join(cell['source'])


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def build():
    assert sha(BASE) == BASE_SHA, 'The scored baseline changed'
    base = json.loads(BASE.read_text(encoding='utf-8'))
    metadata = json.loads((BASE.parent / 'kernel-metadata.json').read_text(encoding='utf-8'))
    RES.mkdir(exist_ok=True)
    plans = []
    for name, (division, readmit, rationale) in CANDIDATES.items():
        folder = W / ('k_e071_' + name.lower())
        assert not (RES / f'{name}_launch.json').exists(), 'Never rebuild a launched notebook'
        nb = copy.deepcopy(base)
        original = text(nb['cells'][0])
        edited = original
        knobs = {'BIOHUB_ILP_DIVISION_WEIGHT': division, 'BIOHUB_READMIT_MIN_SCORE': readmit}
        for key, old in [('BIOHUB_ILP_DIVISION_WEIGHT', '1.2'), ('BIOHUB_READMIT_MIN_SCORE', '0.965')]:
            anchor = f'os.environ["{key}"] = "{old}"'
            assert edited.count(anchor) == 1
            edited = edited.replace(anchor, f'os.environ["{key}"] = "{knobs[key]}"')
        nb['cells'][0]['source'] = edited.splitlines(keepends=True)
        guards = dict(knobs, BIOHUB_VALIDATOR_ENABLE='0', BIOHUB_OUTPUT_MIN_TRACK_LEN='7',
                      BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD='0.25', BIOHUB_READMIT_RADIUS_UM='4',
                      BIOHUB_DET_THRESHOLD='0.965', BIOHUB_REPAIR_DEADLINE_S='34200')
        guard = '\n# E071: preserve the scored baseline except for the declared parameters.\n'
        guard += f'_e071_expected = {guards!r}\n'
        guard += 'for _k, _v in _e071_expected.items():\n    assert _guard_os.environ.get(_k) == _v, (_k, _guard_os.environ.get(_k), _v)\n'
        guard += 'assert _guard_os.environ.get("BIOHUB_XR_FORK_MIN_BRANCH", "0") == "0"\n'
        guard += f'print("E071 CONFIG", {name!r}, _guard_json.dumps(_e071_expected, sort_keys=True), flush=True)\n'
        nb['cells'][1]['source'] = (text(nb['cells'][1]) + guard).splitlines(keepends=True)
        for i in range(2, len(base['cells'])):
            assert text(nb['cells'][i]) == text(base['cells'][i]), ('Unexpected pipeline edit', i)
        for c in nb['cells']:
            if c['cell_type'] == 'code':
                c['outputs'] = []
                c['execution_count'] = None
                ast.parse(text(c))
        source = '\n'.join(text(c) for c in nb['cells'] if c['cell_type'] == 'code')
        manifest = dict(experiment='E071', candidate=name, base='c3_public_0.955',
                        base_submission=56592151, base_notebook_sha256=BASE_SHA,
                        overrides=knobs, rationale=rationale,
                        source_sha256=hashlib.sha256(source.encode()).hexdigest())
        marker = 'import json as _manifest_json\nfrom pathlib import Path as _ManifestPath\n'
        marker += f'_manifest = {manifest!r}\n'
        marker += f'assert ILP_DIVISION_WEIGHT == {float(division)!r}\nassert READMIT_MIN_SCORE == {float(readmit)!r}\n'
        marker += 'assert not VALIDATOR_ENABLE\n'
        marker += '_ManifestPath("/kaggle/working/run_manifest.json").write_text(_manifest_json.dumps(_manifest,sort_keys=True))\n'
        marker += 'print("RUN_MANIFEST", _manifest_json.dumps(_manifest,sort_keys=True), flush=True)\n'
        nb['cells'].append(dict(cell_type='code', metadata={}, outputs=[], execution_count=None,
                                source=marker.splitlines(keepends=True)))
        ast.parse(marker)
        folder.mkdir(exist_ok=True)
        write(folder / 'notebook.ipynb', nb)
        meta = dict(metadata, id='jarturo/biohub-e071-' + name.lower().replace('_', '-'),
                    title='Biohub E071 ' + name.replace('_', ' '))
        assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
        write(folder / 'kernel-metadata.json', meta)
        write(folder / 'expected_manifest.json', manifest)
        diff = '\n'.join(difflib.unified_diff(original.splitlines(), edited.splitlines(), fromfile='c3_cell0', tofile=name))
        (RES / f'{name}.diff').write_text(diff + '\n', encoding='utf-8')
        plans.append(dict(candidate=name, kernel=meta['id'], folder=str(folder),
                          notebook_sha256=sha(folder / 'notebook.ipynb'), overrides=knobs,
                          evidence=rationale, status='BUILT_NOT_EXECUTED'))
    write(RES / 'candidate_plan.json', dict(experiment='E071', candidates=plans,
          auto_submit=False, base_public_score=0.955, max_concurrent_gpu_sessions=2,
          preserve_other_projects=True, reminder_enabled=False))
    print(json.dumps({'built': [p['candidate'] for p in plans], 'pipeline_cells_2_to_11_identical_to_c3': True}))


if __name__ == '__main__':
    build()
