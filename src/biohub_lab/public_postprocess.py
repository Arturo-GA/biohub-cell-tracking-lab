"""Load only frozen public configuration and postprocessing definitions.

No setup, primary inference, test export, or validation sweep is executed.
"""
import types

def load(source):
    prefix=source.split('print("Biohub learned UNet + node-transformer + ILP submission")')[0]
    prefix=prefix.replace('from IPython.display import display','').replace('import pandas as pd','')
    section=source.split('SUBMISSION_COLUMNS = ',1)[1].split('DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()',1)[0]
    section=section.replace('blosc2.decompress(raw)', '__import__("blosc2").decompress(raw)')
    code=prefix+'\nimport numpy as np\nfrom scipy.optimize import linear_sum_assignment\nfrom scipy.spatial import cKDTree\nSUBMISSION_COLUMNS = '+section
    module=types.ModuleType('frozen_public_postprocess')
    exec(compile(code,'frozen_public_postprocess','exec'),module.__dict__)
    return module

def fast_checkpoint_paths(source):
    """Avoid scanning all competition chunks when pinned paths already exist."""
    old='''    if input_root.exists():
        candidates.extend(input_root.rglob("ARTIFACT_MANIFEST.json"))'''
    new='''    _known_good = []
    for _known in candidates:
        try:
            if _known.is_file() and json.loads(_known.read_text()).get("model", {}).get("weight_sha256") == _secondary_expected_sha256:
                _known_good.append(_known)
        except (OSError, ValueError):
            pass
    if _known_good:
        candidates = _known_good
    elif input_root.exists():
        candidates.extend(input_root.rglob("ARTIFACT_MANIFEST.json"))'''
    assert source.count(old)==1;source=source.replace(old,new)
    old='''    if input_root.exists():
        for name in ("checkpoint_last.pt", "best.pt", "last.pt"):'''
    new='''    if input_root.exists() and not (explicit and Path(explicit).is_file()):
        for name in ("checkpoint_last.pt", "best.pt", "last.pt"):'''
    assert source.count(old)==1;source=source.replace(old,new)
    compile(source,'fast_public_paths','exec');return source
