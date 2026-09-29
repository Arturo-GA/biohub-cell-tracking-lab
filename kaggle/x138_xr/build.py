"""Build Kaggle kernels derived from the public 0.953 notebook (anvithpothula/biohub-x138).

k_repro   : exact copy (safety submission)
k_capture : same pipeline run on TRAIN videos in V1284 capture mode + head training
k_infer   : x138 with configurable head(s) / alpha (variants via env block)
"""
import copy
import json
import sys
from pathlib import Path

W = Path(__file__).parent
def _first(*cands):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return Path(cands[0])


X138_NB = _first(W / "x138" / "biohub-x138.ipynb", W / "upstream" / "biohub-x138.ipynb")
OFFICIAL_DIR = _first(W / "r" / "src" / "biohub_official", W.parents[1] / "src" / "biohub_official")
LABROWS = ([p for p in (W / "L0" / "lab" / "lab_rows.csv", W / "L1" / "lab" / "lab_rows.csv") if p.exists()]
           or [p for p in (W.parents[1] / "results" / "E068" / "lab_rows_lab0.csv", W.parents[1] / "results" / "E068" / "lab_rows_lab1.csv") if p.exists()])
SRC = json.loads(X138_NB.read_text(encoding="utf-8"))
DATASETS = [
    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
    "pilkwang/biohub-tracking-support-pack-50ep-v1",
    "anvithpothula/biohub-v1284-head-s075",
]


def clean(nb):
    nb = copy.deepcopy(nb)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
    return nb


def src(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def must_replace(text, old, new, count=1):
    if text.count(old) != count:
        raise SystemExit(f"anchor count {text.count(old)} != {count}: {old[:80]!r}")
    return text.replace(old, new)


def write_kernel(folder, slug, title, nb, datasets=DATASETS, kernel_sources=(), gpu=True):
    d = W / folder
    d.mkdir(exist_ok=True)
    (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {
        "id": f"jarturo/{slug}",
        "title": title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": list(datasets),
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": list(kernel_sources),
        "model_sources": [],
    }
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", d)


# ---------------------------------------------------------------- refine module upgrade
# Adds: V1284_HEADS (comma list, shifts averaged), V1284_ALPHA (shift scale).
OLD_LOAD = (
    "    if _CACHE is None:\\n"
    "        saved = torch.load(os.environ[\\'V1284_HEAD\\'], map_location=\\'cpu\\', weights_only=True)\\n"
    "        head = make_head().to(feature.device)\\n"
    "        head.load_state_dict(saved[\\'state_dict\\']); head.eval()\\n"
    "        _CACHE = (head, saved[\\'mean\\'].to(feature.device), saved[\\'scale\\'].to(feature.device))\\n"
    "    head, mean, scale = _CACHE\\n"
    "    shift = bounded(head, (x-mean)/scale).cpu().numpy() / SPACING\\n"
)
NEW_LOAD = (
    "    if _CACHE is None:\\n"
    "        _CACHE = []\\n"
    "        for _p in (os.environ.get(\\'V1284_HEADS\\') or os.environ[\\'V1284_HEAD\\']).split(\\',\\'):\\n"
    "            saved = torch.load(_p.strip(), map_location=\\'cpu\\', weights_only=True)\\n"
    "            head = make_head().to(feature.device)\\n"
    "            head.load_state_dict(saved[\\'state_dict\\']); head.eval()\\n"
    "            _CACHE.append((head, saved[\\'mean\\'].to(feature.device), saved[\\'scale\\'].to(feature.device)))\\n"
    "        print(\\'V1284 heads loaded:\\', len(_CACHE), \\'alpha=\\', os.environ.get(\\'V1284_ALPHA\\', \\'1\\'), flush=True)\\n"
    "    _alpha = float(os.environ.get(\\'V1284_ALPHA\\', \\'1\\'))\\n"
    "    shift_um = sum(bounded(h, (x-m)/s) for h, m, s in _CACHE) / len(_CACHE)\\n"
    "    shift = (_alpha * shift_um).cpu().numpy() / SPACING\\n"
)
OLD_CHECK = "*SPACING,axis=1)) > 2.00001:"
NEW_CHECK = "*SPACING,axis=1)) > 2.00001*max(1.0, float(os.environ.get(\\'V1284_ALPHA\\', \\'1\\'))):"


def upgrade_refine(cell4_text):
    t = must_replace(cell4_text, OLD_LOAD, NEW_LOAD)
    t = must_replace(t, OLD_CHECK, NEW_CHECK)
    return t


# ---------------------------------------------------------------- 1. exact repro
if __name__ == "__main__" and len(sys.argv) == 1:
    write_kernel("k_repro", "biohub-x138-repro", "Biohub x138 repro", clean(SRC))

# ---------------------------------------------------------------- 2. capture + train
CAPTURE_SELECT = '''
TEST_DIR = COMP_DIR / "train"   # capture runs the unchanged detector on TRAIN videos
'''
CAPTURE_STEMS = '''
import random as _rnd
_all_train = list_test_stems()
_rnd.Random(20260924).shuffle(_all_train)
CAPTURE_N = int(os.environ.get("CAPTURE_N", "110"))
CAPTURE_START = int(os.environ.get("CAPTURE_START", "0"))
test_stems = sorted(_all_train[CAPTURE_START:CAPTURE_START + CAPTURE_N])
'''
CAPTURE_LAUNCH = r'''
# ---- capture launcher: two GPU shards, hard deadline, partial captures are kept
CAPTURE_DEADLINE_S = float(os.environ.get("CAPTURE_DEADLINE_S", str(4.3 * 3600)))
start_time = time.time()
available_gpu_count = _torch.cuda.device_count()
worker_count = max(1, min(2, available_gpu_count))
cuda_tokens = _visible_cuda_tokens(worker_count) if worker_count >= 2 else ["0"]
capture_cmd = [a for a in predict_cmd if a != "--use-ilp"]
processes = {}
for shard_index in range(worker_count):
    shard_cmd = [*capture_cmd, "--method", f"{METHOD}_cap{shard_index}", "--slice", f"{shard_index}::{worker_count}"]
    shard_env = {**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": cuda_tokens[shard_index]}
    print("CAPTURE shard", shard_index, " ".join(shard_cmd), flush=True)
    processes[shard_index] = subprocess.Popen(shard_cmd, cwd=REPO_DIR, env=shard_env)
while any(p.poll() is None for p in processes.values()):
    if time.time() - start_time > CAPTURE_DEADLINE_S:
        print("CAPTURE deadline reached; terminating shards", flush=True)
        for p in processes.values():
            if p.poll() is None:
                p.kill()
        break
    time.sleep(10)
for i, p in processes.items():
    print("shard", i, "return code", p.poll(), flush=True)
_cap = Path(os.environ["V1284_CAPTURE"])
print("captured movies:", len(list(_cap.iterdir())) if _cap.exists() else 0,
      "frames:", len(list(_cap.rglob("*.npz"))) if _cap.exists() else 0,
      f"| {(time.time() - start_time) / 60:.1f} min", flush=True)
'''

def make_training_captures():
    TRAIN_CELL = (W / "train_heads.py").read_text(encoding="utf-8")

    cap = clean(SRC)
    cells = cap["cells"]
    c4 = src(cells[4])
    c4 = must_replace(c4, "def list_test_stems() -> list[str]:", CAPTURE_SELECT + "\ndef list_test_stems() -> list[str]:")
    c4 = must_replace(c4, "test_stems = list_test_stems()\n", CAPTURE_STEMS)
    c4 = must_replace(c4, "os.environ['V1284_MODE']='candidate'",
                      "os.environ['V1284_MODE']='capture'\nos.environ['V1284_CAPTURE']='/kaggle/working/capture'")
    cut = c4.index("start_time = time.time()\navailable_gpu_count")
    c4 = c4[:cut] + CAPTURE_LAUNCH
    set_src(cells[4], c4)
    train_cell = copy.deepcopy(cells[5])
    set_src(train_cell, TRAIN_CELL)
    cap["cells"] = cells[:5] + [train_cell]
    write_kernel("k_capture", "biohub-v1284-capture-train", "Biohub V1284 capture train", cap)
    cap2 = copy.deepcopy(cap)
    set_src(cap2["cells"][0], 'import os\nos.environ["CAPTURE_START"] = "110"\nos.environ["CAPTURE_N"] = "200"\n'
            'os.environ["HEAD_SAVE_PAIRS"] = "1"\n' + src(cap2["cells"][0]))
    write_kernel("k_capture2", "biohub-v1284-capture-train2", "Biohub V1284 capture train2", cap2)



# ---------------------------------------------------------------- 3. inference variants
def make_infer(folder, slug, title, heads_expr=None, alpha=None, kernel_sources=()):
    nb = clean(SRC)
    cells = nb["cells"]
    c4 = upgrade_refine(src(cells[4]))
    extra = ""
    if heads_expr is not None:
        extra += (
            "_new_heads = sorted(str(p) for p in Path('/kaggle/input').rglob('" + heads_expr + "'))\n"
            "if not _new_heads:\n    raise RuntimeError('no new V1284 heads found')\n"
            "os.environ['V1284_HEADS'] = ','.join(_new_heads)\n"
            "print('V1284_HEADS =', _new_heads)\n"
        )
    if alpha is not None:
        extra += f"os.environ['V1284_ALPHA'] = '{alpha}'\n"
    c4 = must_replace(c4, "os.environ['V1284_HEAD']=str(_myhead[0])\n",
                      "os.environ['V1284_HEAD']=str(_myhead[0])\n" + extra)
    set_src(cells[4], c4)
    write_kernel(folder, slug, title, nb, kernel_sources=kernel_sources)


# ---------------------------------------------------------------- 4. Level 1: 10 heads + validator-driven strict post-process sweep
OLD_CANDS = '''PP_CANDIDATES: dict[str, dict] = {
    "gap45": {"GAP_CLOSE_UM": 4.5},
    "tight55": {"MOTION_RELINK_TIGHT_UM": 5.5},
    "relaxed9": {"MOTION_RELINK_RELAXED_UM": 9.0},
    "bonus125": {"MOTION_RELINK_LEARNED_BONUS": 1.25},
    "gap2step40": {"GAP2_MAX_STEP_UM": 4.0},
    "reuse28": {"GAP_CLOSE_REUSE_UM": 2.8},
    "dcgap035": {"DEEPCENTER_GAP_THRESHOLD": 0.35},
}'''
NEW_CANDS = '''# Level 1: the base thresholds were tuned on integer-grid coordinates (~1.45 um
# quantisation noise). With V1284 refinement the coordinates are ~1.05 um from
# GT, so stricter um-tolerances and a stronger velocity prior are re-tested.
PP_CANDIDATES: dict[str, dict] = {
    "tight50": {"MOTION_RELINK_TIGHT_UM": 5.0},
    "tight45": {"MOTION_RELINK_TIGHT_UM": 4.5},
    "relaxed9": {"MOTION_RELINK_RELAXED_UM": 9.0},
    "vel07": {"MOTION_RELINK_VELOCITY_WEIGHT": 0.7},
    "vel03": {"MOTION_RELINK_VELOCITY_WEIGHT": 0.3},
    "bonus125": {"MOTION_RELINK_LEARNED_BONUS": 1.25},
    "gap45": {"GAP_CLOSE_UM": 4.5},
    "gap55": {"GAP_CLOSE_UM": 5.5},
    "gap2step40": {"GAP2_MAX_STEP_UM": 4.0},
    "gap2total9": {"GAP2_MAX_TOTAL_UM": 9.0},
    "reuse28": {"GAP_CLOSE_REUSE_UM": 2.8},
    "diverge20": {"SAFE_DIV_DIVERGE_UM": 2.0},
    "diverge25": {"SAFE_DIV_DIVERGE_UM": 2.5},
    "sister13": {"SAFE_DIV_SISTER_MAX_UM": 13.0},
    "dcgap035": {"DEEPCENTER_GAP_THRESHOLD": 0.35},
    "minlen5": {"OUTPUT_MIN_TRACK_LEN": 5},
    "minlen7": {"OUTPUT_MIN_TRACK_LEN": 7},
}'''


def make_level1(folder, slug, title, own_heads=False):
    nb = clean(SRC)
    cells = nb["cells"]
    c0 = must_replace(src(cells[0]), 'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "0"', 'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "1"')
    set_src(cells[0], c0)
    c4 = src(cells[4])
    sources = []
    if own_heads:
        # LB 2026-09-24: own 5-head ensemble scored 0.952 vs public head 0.953 -> public head is the base.
        c4 = upgrade_refine(c4)
        extra = (
            "_new_heads = sorted(str(p) for p in Path('/kaggle/input').rglob('v1284_ours_s*.pt'))\n"
            "if len(_new_heads) != 10:\n    raise RuntimeError(('expected 10 V1284 heads', _new_heads))\n"
            "os.environ['V1284_HEADS'] = ','.join(_new_heads)\n"
            "print('V1284_HEADS =', _new_heads)\n"
        )
        c4 = must_replace(c4, "os.environ['V1284_HEAD']=str(_myhead[0])\n", "os.environ['V1284_HEAD']=str(_myhead[0])\n" + extra)
        sources = ["jarturo/biohub-v1284-capture-train", "jarturo/biohub-v1284-capture-train2"]
    set_src(cells[4], c4)
    set_src(cells[10], must_replace(src(cells[10]), OLD_CANDS, NEW_CANDS))
    write_kernel(folder, slug, title, nb, kernel_sources=sources)


def make_fixed(folder, slug, title, overrides):
    """x138 exactly, with post-process env overrides hardcoded in cell 0.
    No validator, no sweep, no rewrite: hidden-rerun runtime identical to x138."""
    nb = clean(SRC)
    cells = nb["cells"]
    c0 = src(cells[0])
    for key, value in overrides.items():
        line_old = [l for l in c0.splitlines() if l.startswith(f'os.environ["{key}"]')]
        if len(line_old) != 1:
            raise SystemExit(f"{key}: expected exactly one assignment in cell 0, found {len(line_old)}")
        c0 = must_replace(c0, line_old[0], f'os.environ["{key}"] = "{value}"  # fixed override')
    set_src(cells[0], c0)
    c1 = src(cells[1])
    for key, value in overrides.items():
        old = f'    "{key}": '
        if old in c1:
            line = [l for l in c1.splitlines() if l.startswith(old)][0]
            c1 = must_replace(c1, line, f'    "{key}": {float(value)},')
    set_src(cells[1], c1)
    write_kernel(folder, slug, title, nb)


LAB_VAL8 = ['44b6_12dfb391', '44b6_267148e4', '44b6_2a2eff9f', '44b6_341df25f',
            '6bba_062c8d37', '6bba_07e24132', '6bba_085bf656', '6bba_09961292']
LAB_SELECT = '''
TEST_DIR = COMP_DIR / "train"   # pruning lab: unchanged x138 pipeline on held-out TRAIN videos
'''
LAB_STEMS = '''
import random as _rnd
_all_train = list_test_stems()
_val8 = %r
_by_pfx = {}
for _s in _all_train:
    _by_pfx.setdefault(_s.split("_")[0], []).append(_s)
LAB_PART = int(os.environ.get("LAB_PART", "0"))
_done72 = set(%r)
_sel = []
if LAB_PART == 5:
    # division lab: the 36 train videos with the most annotated divisions (GT read with tracksdata)
    import collections as _col
    import tracksdata as _td
    _counts = {}
    for _s in _all_train:
        try:
            _g = _td.graph.IndexedRXGraph.from_geff(str(COMP_DIR / "train" / f"{_s}.geff"))
            _g = _g[0] if isinstance(_g, tuple) else _g
            _c = _col.Counter(_g.edge_attrs()["source_id"].to_list())
            _counts[_s] = sum(1 for _v in _c.values() if _v >= 2)
        except Exception as _exc:
            print("GT count failed", _s, type(_exc).__name__, _exc)
            _counts[_s] = -1
    _sel = sorted(_all_train, key=lambda s: (-_counts[s], s))[:36]
    print("GT divisions total:", sum(v for v in _counts.values() if v > 0), "| selected:", [(s, _counts[s]) for s in _sel])
elif LAB_PART >= 6:
    # parts 6,7,8: train videos without a safe-div trace yet (lab4/lab5 attached as kernel sources), 3 interleaved parts
    import numpy as _np
    from pathlib import Path as _P
    _traced = set()
    for _p in _P("/kaggle/input").rglob("lab/graphs/*.npz"):
        try:
            if "sdtrace" in _np.load(_p).files:
                _traced.add(_p.stem)
        except Exception:
            pass
    _rem = sorted(s for s in _all_train if s not in _traced)
    _sel = _rem[(LAB_PART - 6)::3]
    print("already traced:", len(_traced), "| remaining:", len(_rem))
elif LAB_PART >= 2:
    # parts 2,3,4: the 127 train videos not captured by lab0/lab1, interleaved so each part mixes both embryos
    _rem = sorted(s for s in _all_train if s not in _done72)
    _sel = _rem[(LAB_PART - 2)::3]
else:
    for _pfx, _ss in sorted(_by_pfx.items()):
        _ss = sorted(_ss)
        _rnd.Random(7).shuffle(_ss)
        _others = [s for s in _ss if s not in _val8]
        if LAB_PART == 0:
            _sel += [s for s in _val8 if s.startswith(_pfx)] + _others[:14]
        else:
            _sel += _others[14:32]
test_stems = sorted(_sel)
print("LAB stems:", len(test_stems), test_stems)
''' % (LAB_VAL8, sorted((W / "done72.txt").read_text().split()))


def make_lab(folder, slug, title, part, trace=False, sources=()):
    nb = clean(SRC)
    cells = nb["cells"]
    c0 = 'import os\nos.environ["LAB_PART"] = "%d"\n' % part + src(cells[0])
    set_src(cells[0], c0)
    c4 = src(cells[4])
    c4 = must_replace(c4, "def list_test_stems() -> list[str]:", LAB_SELECT + "\ndef list_test_stems() -> list[str]:")
    c4 = must_replace(c4, "test_stems = list_test_stems()\n", LAB_STEMS)
    set_src(cells[4], c4)
    c5 = src(cells[5])
    capture_src = (W / "lab_capture.py").read_text(encoding="utf-8")
    if trace:
        capture_src = (W / "trace_safediv.py").read_text(encoding="utf-8") + "\n" + capture_src
    c5 = must_replace(c5, '\nwrite_test_submission("base")', "\n" + capture_src)
    set_src(cells[5], c5)
    lab_cell = copy.deepcopy(cells[8])
    set_src(lab_cell, (W / "lab_eval.py").read_text(encoding="utf-8"))
    nb["cells"] = cells[:6] + [cells[8], lab_cell]
    write_kernel(folder, slug, title, nb, kernel_sources=list(sources))


def make_cpulab(folder, slug, title):
    """CPU-only analysis kernel over the lab0/lab1 captured graphs (no GPU quota)."""
    nb = clean(SRC)
    cells = nb["cells"]
    c5 = src(cells[5])
    a = c5.index("def filter_short_track_components(")
    b = c5.index("\n\ndef linefit_smooth_output_graph(")
    filter_src = c5[a:b]
    off = {}
    for name in ("__init__.py", "metrics.py", "division_metrics.py"):
        off[name] = (OFFICIAL_DIR / name).read_text(encoding="utf-8")
    ref = {}
    import csv as _csv
    for p in LABROWS:
        for r in _csv.DictReader(p.open(encoding="utf-8")):
            if r["rule"] == "L7":
                ref[r["stem"]] = float(r["adjusted_edge_jaccard"])
    body = (W / "cpu_lab.py").read_text(encoding="utf-8")
    body = body.replace("__LAB_L7_REF__", repr(ref)).replace("__OFFICIAL_FILES__", repr(off)).replace("__FILTER_SRC__", repr(filter_src))
    pre = copy.deepcopy(cells[8])
    set_src(pre, "import json\nimport math\nfrom pathlib import Path\n\nimport numpy as np\n\nVOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)\n")
    lab_cell = copy.deepcopy(cells[8])
    set_src(lab_cell, body)
    nb["cells"] = [cells[0], cells[2], cells[3], pre, cells[8], lab_cell]
    write_kernel(folder, slug, title, nb, kernel_sources=["jarturo/biohub-prune-lab0", "jarturo/biohub-prune-lab1"], gpu=False)


def _lab_embed_common():
    c5 = src(SRC["cells"][5])
    a = c5.index("def filter_short_track_components(")
    b = c5.index("\n\ndef linefit_smooth_output_graph(")
    off = {name: (OFFICIAL_DIR / name).read_text(encoding="utf-8")
           for name in ("__init__.py", "metrics.py", "division_metrics.py")}
    ref = {}
    import csv as _csv
    for p in LABROWS:
        for r in _csv.DictReader(p.open(encoding="utf-8")):
            if r["rule"] == "L7":
                ref[r["stem"]] = float(r["adjusted_edge_jaccard"])
    return c5[a:b], off, ref


def make_cpulab2(folder, slug, title, sources):
    nb = clean(SRC)
    cells = nb["cells"]
    filter_src, off, ref = _lab_embed_common()
    body = (W / "cpu_lab2.py").read_text(encoding="utf-8")
    body = (body.replace("__LAB_L7_REF__", repr(ref)).replace("__OFFICIAL_FILES__", repr(off))
            .replace("__FILTER_SRC__", repr(filter_src)).replace("__XR_SRC__", repr((W / "extra_rules.py").read_text(encoding="utf-8"))))
    pre = copy.deepcopy(cells[8])
    set_src(pre, "import json\nimport math\nfrom pathlib import Path\n\nimport numpy as np\n\nVOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)\n")
    lab_cell = copy.deepcopy(cells[8])
    set_src(lab_cell, body)
    nb["cells"] = [cells[0], cells[2], cells[3], pre, lab_cell]
    write_kernel(folder, slug, title, nb, kernel_sources=list(sources), gpu=False)


XR_OLD_CALL = "    nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats)\n"
XR_NEW_CALL = (
    "    nodes_by_id, edges = xr_pre_filter(nodes_by_id, edges, dataset, stats)\n"
    "    nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats)\n"
    "    nodes_by_id, edges = xr_post_filter(nodes_by_id, edges, dataset, stats)\n"
)


def make_xr(folder, slug, title, overrides, xr_env):
    """x138 single pass + fixed overrides + the shared XR rule module (env BIOHUB_XR_* set in cell 0)."""
    make_fixed(folder, slug, title, overrides)
    d = W / folder
    nb = json.loads((d / "notebook.ipynb").read_text(encoding="utf-8"))
    cells = nb["cells"]
    c0 = src(cells[0])
    anchor = 'print("BIOHUB_PRESET:", BIOHUB_PRESET)'
    lines = "".join(f'os.environ["{k}"] = "{v}"  # XR rule\n' for k, v in xr_env.items())
    set_src(cells[0], must_replace(c0, anchor, lines + anchor))
    c5 = src(cells[5])
    c5 = must_replace(c5, "def filter_output_graph(", (W / "extra_rules.py").read_text(encoding="utf-8") + "\ndef filter_output_graph(")
    c5 = must_replace(c5, XR_OLD_CALL, XR_NEW_CALL)
    c5 = must_replace(c5, "    edges = add_safe_divisions_postlink(\n        nodes_by_id,\n        edges,\n        stats,\n        dataset=dataset,",
                      "    xr_set_safe_div_params(dataset)\n    edges = add_safe_divisions_postlink(\n        nodes_by_id,\n        edges,\n        stats,\n        dataset=dataset,")
    set_src(cells[5], c5)
    (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("XR patched", folder, xr_env)


def make_cpulab3(folder, slug, title, sources, lab_file="cpu_lab3.py"):
    nb = clean(SRC)
    cells = nb["cells"]
    filter_src, off, ref = _lab_embed_common()
    body = (W / lab_file).read_text(encoding="utf-8")
    body = (body.replace("__OFFICIAL_FILES__", repr(off)).replace("__FILTER_SRC__", repr(filter_src))
            .replace("__XR_SRC__", repr((W / "extra_rules.py").read_text(encoding="utf-8"))))
    if lab_file in ("cpu_lab4.py", "cpu_lab5.py", "cpu_lab6.py", "cpu_lab7.py"):
        import ast
        c5 = src(cells[5])
        linefit = next(ast.get_source_segment(c5, n) for n in ast.parse(c5).body
                       if isinstance(n, ast.FunctionDef) and n.name == "linefit_smooth_output_graph")
        body = body.replace("__LINEFIT_SRC__", repr(linefit)).replace(
            "__REPLAY_SRC__", repr((W / "replay_linefit.py").read_text(encoding="utf-8")))
    if lab_file == "cpu_lab5.py":
        body = body.replace("__LINEAGE_SRC__", repr((W / "lineage_repair.py").read_text(encoding="utf-8")))
    if lab_file == "cpu_lab6.py":
        body = body.replace("__CONTEXT_SRC__", repr((W / "context_rules.py").read_text(encoding="utf-8")))
    if lab_file == "cpu_lab7.py":
        body = body.replace("__C3_VARIANTS_SRC__", repr((W / "c3_variants.py").read_text(encoding="utf-8")))
    pre = copy.deepcopy(cells[8])
    set_src(pre, "import json\nimport math\nfrom pathlib import Path\n\nimport numpy as np\n\nVOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)\n")
    lab_cell = copy.deepcopy(cells[8])
    set_src(lab_cell, body)
    nb["cells"] = [cells[0], cells[2], cells[3], pre, lab_cell]
    write_kernel(folder, slug, title, nb, kernel_sources=list(sources), gpu=False)


if __name__ == "__main__" and "--cpulab4" in sys.argv:
    i = sys.argv.index("--cpulab4")
    make_cpulab3("k_cpu4", "biohub-exact-replay-cpu4", "Biohub exact replay cpu4",
                sys.argv[i + 1].split(","), lab_file="cpu_lab4.py")
    sys.exit(0)

if __name__ == "__main__" and "--cpulab3" in sys.argv:
    i = sys.argv.index("--cpulab3")
    make_cpulab3("k_cpu3", "biohub-safediv-lab-cpu3", "Biohub safediv lab cpu3", sys.argv[i + 1].split(","))
    sys.exit(0)

if __name__ == "__main__" and "--cpulab2" in sys.argv:
    i = sys.argv.index("--cpulab2")
    srcs = sys.argv[i + 1].split(",")
    make_cpulab2("k_cpu2", "biohub-prune-lab-cpu2", "Biohub prune lab cpu2", srcs)
    sys.exit(0)

if __name__ == "__main__" and "--xr" in sys.argv:
    # usage: python build.py --xr folder slug "title" KEY=VAL ... -- BIOHUB_XR_KEY=VAL ...
    i = sys.argv.index("--xr")
    folder, slug, title = sys.argv[i + 1:i + 4]
    rest = sys.argv[i + 4:]
    cut = rest.index("--") if "--" in rest else len(rest)
    make_xr(folder, slug, title, dict(kv.split("=", 1) for kv in rest[:cut]), dict(kv.split("=", 1) for kv in rest[cut + 1:]))
    sys.exit(0)

if __name__ == "__main__" and "--cpulab" in sys.argv:
    make_cpulab("k_cpu", "biohub-prune-lab-cpu", "Biohub prune lab cpu")
    sys.exit(0)

if __name__ == "__main__" and "--lab" in sys.argv:
    make_lab("k_lab0", "biohub-prune-lab0", "Biohub prune lab0", 0)
    make_lab("k_lab1", "biohub-prune-lab1", "Biohub prune lab1", 1)
    for part in (2, 3):
        make_lab(f"k_lab{part}", f"biohub-prune-lab{part}", f"Biohub prune lab{part}", part)
    make_lab("k_lab4", "biohub-prune-lab4", "Biohub prune lab4", 4, trace=True)
    make_lab("k_lab5", "biohub-prune-lab5", "Biohub prune lab5", 5, trace=True)   # title must slugify to the id
    for part in (6, 7, 8):
        make_lab(f"k_lab{part}", f"biohub-prune-lab{part}", f"Biohub prune lab{part}", part, trace=True,
                 sources=["jarturo/biohub-prune-lab4", "jarturo/biohub-prune-lab5"])
    sys.exit(0)

OLD_SHORT_CALL = "    nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats)\n"
NEW_SHORT_CALL = (
    "    _ts_all = [int(_n[\"t\"]) for _n in nodes_by_id.values()]\n"
    "    nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats)\n"
    "    if _ts_all:\n"
    "        nodes_by_id, edges = extra_prune_components(nodes_by_id, edges, stats, min(_ts_all), max(_ts_all))\n"
)


def make_extra(folder, slug, title, overrides, new_env):
    """x138 single pass + fixed overrides + optional extra pruning stage (new env keys appended to cell 0)."""
    make_fixed(folder, slug, title, overrides)
    d = W / folder
    nb = json.loads((d / "notebook.ipynb").read_text(encoding="utf-8"))
    cells = nb["cells"]
    c0 = src(cells[0])
    anchor = 'print("BIOHUB_PRESET:", BIOHUB_PRESET)'
    lines = "".join(f'os.environ["{k}"] = "{v}"  # extra prune\n' for k, v in new_env.items())
    c0 = must_replace(c0, anchor, lines + anchor)
    set_src(cells[0], c0)
    c5 = src(cells[5])
    c5 = must_replace(c5, "def filter_output_graph(", (W / "extra_prune.py").read_text(encoding="utf-8") + "\ndef filter_output_graph(")
    c5 = must_replace(c5, OLD_SHORT_CALL, NEW_SHORT_CALL)
    set_src(cells[5], c5)
    (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("extra-prune patched", folder, new_env)


if __name__ == "__main__" and "--extra" in sys.argv:
    # usage: python build.py --extra folder slug "title" KEY=VAL ... -- NEWKEY=VAL ...
    i = sys.argv.index("--extra")
    folder, slug, title = sys.argv[i + 1:i + 4]
    rest = sys.argv[i + 4:]
    cut = rest.index("--") if "--" in rest else len(rest)
    ov = dict(kv.split("=", 1) for kv in rest[:cut])
    ne = dict(kv.split("=", 1) for kv in rest[cut + 1:])
    make_extra(folder, slug, title, ov, ne)
    sys.exit(0)

if __name__ == "__main__" and "--fixed" in sys.argv:
    # usage: python build.py --fixed folder slug "title" KEY=VAL [KEY=VAL ...]
    i = sys.argv.index("--fixed")
    folder, slug, title = sys.argv[i + 1:i + 4]
    ov = dict(kv.split("=", 1) for kv in sys.argv[i + 4:])
    make_fixed(folder, slug, title, ov)
    sys.exit(0)

if __name__ == "__main__" and "--level1" in sys.argv:
    make_level1("k_level1", "biohub-x138-level1", "Biohub x138 level1", own_heads=False)
    sys.exit(0)

if __name__ == "__main__":
    for spec in sys.argv[1:]:
        # folder|slug|title|heads_glob|alpha|kernel_source
        folder, slug, title, heads, alpha, ks = (spec.split("|") + [""] * 6)[:6]
        make_infer(folder, slug, title, heads or None, alpha or None, [ks] if ks else [])
