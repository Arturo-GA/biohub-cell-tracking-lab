# ---- CPU LAB 3: replay the safe-division decision with other thresholds, using the candidate trace
# (sdtrace) recorded by the traced GPU labs, and score every variant with the OFFICIAL metric.
import collections
import csv as _csv
import json as _json
import sys
import time as _t
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

OUT = WORKING_DIR / "cpu3"
OUT.mkdir(parents=True, exist_ok=True)
T0 = _t.time()
VOX = np.array(VOXEL_SCALE_UM, dtype=np.float64)


def log(*a):
    print(f"[{_t.time() - T0:6.0f}s]", *a, flush=True)


OFF = WORKING_DIR / "offmetric" / "biohub_official"
OFF.mkdir(parents=True, exist_ok=True)
for _name, _text in __OFFICIAL_FILES__.items():
    (OFF / _name).write_text(_text, encoding="utf-8")
sys.path.insert(0, str(OFF.parent))
import polars as pl
import tracksdata as td
from biohub_official.metrics import evaluate as off_evaluate, per_sample_metrics as off_per_sample

exec(__FILTER_SRC__, globals())
exec(__XR_SRC__, globals())

COLS = ["t", "source_id", "child_id", "cand_id", "child_dist", "parent_dist", "sister_dist", "mutual_nn",
        "div_ok", "diverge", "symmetry", "dc_cand", "dc_child", "added"]
CI = {c: i for i, c in enumerate(COLS)}
GRAPHS = {}
for p in sorted(Path("/kaggle/input").rglob("lab/graphs/*.npz")):
    z = np.load(p)
    if "sdtrace" not in z.files or "eflags" not in z.files:
        continue
    if p.stem in GRAPHS:
        continue
    GRAPHS[p.stem] = {k: z[k] for k in z.files}
STEMS = sorted(GRAPHS)
log("traced graphs:", len(STEMS), collections.Counter(s.split("_")[0] for s in STEMS),
    "| candidates:", sum(int(GRAPHS[s]["sdtrace"].shape[0]) for s in STEMS), "| x138-added:", sum(int(GRAPHS[s]["sdtrace"][:, CI["added"]].sum()) for s in STEMS))
TRAIN_DIR = COMP_DIR / "train"


def graph_dicts(stem):
    g = GRAPHS[stem]
    nid = g["nid"].astype(int)
    raw = g["nodes"].astype(np.float64)
    nodes = {int(i): {"node_id": int(i), "t": int(r[0]), "z": float(r[1]), "y": float(r[2]), "x": float(r[3])} for i, r in zip(nid, raw)}
    pos = {int(i): r[1:] * VOX for i, r in zip(nid, raw)}
    pre, sd = [], []
    for (s, d), p, f in zip(g["edges"].astype(int), g["eprob"].astype(np.float64), g["eflags"].astype(int)):
        s, d = int(s), int(d)
        if s not in nodes or d not in nodes:
            continue
        e = {"source_id": s, "target_id": d, "edge_prob": (None if not np.isfinite(p) else float(p)),
             "distance_um": float(np.linalg.norm(pos[s] - pos[d]))}
        (sd if f & 32 else pre).append(e)
    final = {int(i): (int(r[0]), float(r[1]), float(r[2]), float(r[3])) for i, r in zip(nid, g["nodes_final"])}
    t_true = float(g["t_true"][0])
    # per-frame source counts (nodes with exactly one outgoing edge in the pre-safe-div graph) for the frame cap
    out = collections.Counter(e["source_id"] for e in pre)
    n_src = collections.Counter(nodes[n]["t"] for n, c in out.items() if c == 1)
    return nodes, pre, sd, final, (t_true if t_true > 0 else None), n_src


CACHE = {s: graph_dicts(s) for s in STEMS}
log("dict graphs built | x138 safe-div edges in captures:", sum(len(CACHE[s][2]) for s in STEMS))

BASE = dict(child_max=10.0, parent_max=9.0, sister_max=14.0, mutual=True, require_div=True, diverge=2.25,
            dc_thr=0.25, tau=0.6, frame_cap=0.0076, global_cap=0.00375)


def select(stem, cfg):
    tr = GRAPHS[stem]["sdtrace"]
    if tr.shape[0] == 0:
        return []
    nodes, pre, sd, final, t_true, n_src = CACHE[stem]
    m = (tr[:, CI["child_dist"]] <= cfg["child_max"]) & (tr[:, CI["parent_dist"]] <= cfg["parent_max"]) & (tr[:, CI["sister_dist"]] <= cfg["sister_max"])
    if cfg["mutual"]:
        m &= tr[:, CI["mutual_nn"]] == 1
    if cfg["require_div"]:
        m &= (tr[:, CI["div_ok"]] == 1) & (np.nan_to_num(tr[:, CI["diverge"]], nan=-1e9) >= cfg["diverge"])
    if cfg["dc_thr"] > 0:
        dc = tr[:, CI["dc_cand"]]
        m &= ~(np.isfinite(dc) & (dc < cfg["dc_thr"]))
    tau = cfg["tau"]
    if isinstance(tau, dict):
        tau = tau.get(stem.split("_")[0], tau.get("*", 0.6))
    if tau > 0:
        m &= tr[:, CI["symmetry"]] <= tau
    rows = tr[m]
    if rows.shape[0] == 0:
        return []
    global_cap = max(1, int(round(max(1, len(pre)) * cfg["global_cap"]))) if cfg["global_cap"] > 0 else 10 ** 9
    score = rows[:, CI["parent_dist"]] + 0.15 * rows[:, CI["sister_dist"]]
    order = np.lexsort((rows[:, CI["cand_id"]], rows[:, CI["source_id"]], score, rows[:, CI["t"]]))
    rows = rows[order]
    added, used_t, used_s = [], set(), set()
    cur_t, added_this_frame, frame_cap = None, 0, 0
    for r in rows:
        t = int(r[CI["t"]])
        if t != cur_t:
            cur_t, added_this_frame = t, 0
            frame_cap = max(1, int(round(n_src.get(t, 0) * cfg["frame_cap"]))) if cfg["frame_cap"] > 0 else 10 ** 9
        if len(added) >= global_cap or added_this_frame >= frame_cap:
            continue
        s, c = int(r[CI["source_id"]]), int(r[CI["cand_id"]])
        if c in used_t or s in used_s:
            continue
        added.append({"source_id": s, "target_id": c, "edge_prob": None, "distance_um": float(r[CI["parent_dist"]]), "safe_division": 1})
        used_t.add(c); used_s.add(s); added_this_frame += 1
    return added


# ---- self-check: base thresholds must reproduce what x138 added
mis = 0
for s in STEMS:
    got = {(e["source_id"], e["target_id"]) for e in select(s, BASE)}
    ref = {(int(r[CI["source_id"]]), int(r[CI["cand_id"]])) for r in GRAPHS[s]["sdtrace"] if r[CI["added"]] == 1}
    cap = {(e["source_id"], e["target_id"]) for e in CACHE[s][2]}
    if got != ref or ref != cap:
        mis += 1
        if mis <= 5:
            log(f"self-check {s}: replay={len(got)} trace_added={len(ref)} captured_flag32={len(cap)} | replay-ref={len(got - ref)} ref-replay={len(ref - got)} ref-cap={len(ref ^ cap)}")
log("SELF-CHECK safe-div replay: stems with mismatch =", mis, "of", len(STEMS))


def run_filter(nodes, edges, L=7):
    g = globals()
    saved = g["OUTPUT_MIN_TRACK_LEN"]
    g["OUTPUT_MIN_TRACK_LEN"] = int(L)
    try:
        return filter_short_track_components(nodes, edges, collections.Counter())
    finally:
        g["OUTPUT_MIN_TRACK_LEN"] = saved


XR_BEST = {"prune": "*:8:0.7", "cut_nan_dist_um": 8.0, "cut_end_prob": 0.5}


def pipeline(stem, cfg, xr=None):
    nodes, pre, sd, final, t_true, n_src = CACHE[stem]
    edges = [dict(e) for e in pre] + select(stem, cfg)
    xr_configure(cut_nan_dist_um=0.0, cut_end_prob=0.0, cut_end_mode="both", fork_min_branch=0, prune="", cut_end_iter=1, fork_nan_only="0")
    if xr:
        xr_configure(**xr)
    st = {}
    n1, e1 = xr_pre_filter(nodes, edges, stem, st)
    n2, e2 = run_filter(n1, e1, 7)
    n3, e3 = xr_post_filter(n2, e2, stem, st)
    return n3, e3


def build_td(plain, pairs):
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, 0.0)
    keys = sorted(plain)
    ids = graph.bulk_add_nodes([{"t": int(plain[k][0]), "z": float(plain[k][1]), "y": float(plain[k][2]), "x": float(plain[k][3])} for k in keys])
    mapping = dict(zip(keys, ids))
    if pairs:
        graph.bulk_add_edges([{"source_id": mapping[s], "target_id": mapping[d]} for s, d in pairs])
    return graph


def worker(args):
    stem, plain, pairs, t_true = args
    try:
        graph = build_td(plain, pairs)
        truth = td.graph.IndexedRXGraph.from_geff(str(TRAIN_DIR / f"{stem}.geff"))[0]
        res = off_evaluate(graph, truth, scale=tuple(VOXEL_SCALE_UM), max_distance=7.0)
        row = off_per_sample(res, t_true if t_true else float("nan"), float("nan"))
        row["stem"] = stem
        return stem, row, None
    except Exception as exc:
        return stem, None, f"{type(exc).__name__}: {exc}"


def official_rows(graphs, procs=4):
    tasks = [(s, {k: CACHE[s][3][k] for k in graphs[s][0]}, [(int(e["source_id"]), int(e["target_id"])) for e in graphs[s][1]], CACHE[s][4]) for s in STEMS]
    rows = {}
    with Pool(procs) as pool:
        for stem, row, err in pool.imap_unordered(worker, tasks):
            if err:
                log("official error", stem, err)
            else:
                rows[stem] = row
    return rows


def score(rows, stems):
    rs = [rows[s] for s in stems if s in rows]
    w = sum(r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rs) or 1.0
    adj = sum(r["adj_edge_jaccard"] * (r["edge_tp"] + r["edge_fp"] + r["edge_fn"]) for r in rs) / w
    tp = sum(r["division_tp"] for r in rs); fp = sum(r["division_fp"] for r in rs); fn = sum(r["division_fn"] for r in rs)
    dj = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return adj + 0.1 * dj, adj, (tp, fp, fn)


def boot(base, rows, stems, B=1500, seed=0):
    rng = np.random.RandomState(seed)
    d0 = score(rows, stems)[0] - score(base, stems)[0]
    ds = np.sort([score(rows, samp)[0] - score(base, samp)[0]
                  for samp in ([stems[i] for i in rng.randint(0, len(stems), len(stems))] for _ in range(B))])
    win = sum(1 for s in stems if score(rows, [s])[0] > score(base, [s])[0] + 1e-12)
    lose = sum(1 for s in stems if score(rows, [s])[0] < score(base, [s])[0] - 1e-12)
    return d0, ds[int(0.05 * B)], ds[int(0.95 * B)], float((ds > 0).mean()), win, lose


def V(**kw):
    d = dict(BASE); d.update(kw); return d


XR_EL = {"prune": "44b6:8:0.7,6bba:11:0.7", "cut_nan_dist_um": 8.0, "cut_end_prob": 0.5, "cut_end_mode": "last"}
XR_ELF = dict(XR_EL, fork_min_branch=8, fork_nan_only="1")
CONFIGS = [
    ("base(x138)", BASE, None),
    ("tau0.8", V(tau=0.8), None),
    ("tau0.9", V(tau=0.9), None),
    ("tau1.0", V(tau=1.0), None),
    ("tau1.2", V(tau=1.2), None),
    ("6bba0.9|44b6 0.6", V(tau={"44b6": 0.6, "6bba": 0.9}), None),
    ("6bba1.0|44b6 0.6", V(tau={"44b6": 0.6, "6bba": 1.0}), None),
    ("6bba1.2|44b6 0.6", V(tau={"44b6": 0.6, "6bba": 1.2}), None),
    ("6bba1.0|44b6 0.8", V(tau={"44b6": 0.8, "6bba": 1.0}), None),
    ("6bba1.0|44b6 0.9", V(tau={"44b6": 0.9, "6bba": 1.0}), None),
    ("6bba1.0|44b6 1.2", V(tau={"44b6": 1.2, "6bba": 1.0}), None),
    ("tau1.0+parent12", V(tau=1.0, parent_max=12.0), None),
    ("tau0.9+parent12", V(tau=0.9, parent_max=12.0), None),
    ("tau1.0+dc0.28", V(tau=1.0, dc_thr=0.28), None),
    ("base+XR_ELF", BASE, XR_ELF),
    ("6bba1.0|44b6 0.6+XR_ELF", V(tau={"44b6": 0.6, "6bba": 1.0}), XR_ELF),
    ("tau1.0+XR_ELF", V(tau=1.0), XR_ELF),
    ("6bba0.9|44b6 0.6+XR_ELF", V(tau={"44b6": 0.6, "6bba": 0.9}), XR_ELF),
    ("6bba1.0|44b6 0.9+XR_ELF", V(tau={"44b6": 0.9, "6bba": 1.0}), XR_ELF),
    ("tau1.0+parent12+XR_ELF", V(tau=1.0, parent_max=12.0), XR_ELF),
]

ROWS, results, base_rows = {}, [], None
for name, cfg, xr in CONFIGS:
    t1 = _t.time()
    graphs = {s: pipeline(s, cfg, xr) for s in STEMS}
    n_add = sum(len(select(s, cfg)) for s in STEMS)
    rows = official_rows(graphs)
    ROWS[name] = rows
    if base_rows is None:
        base_rows = rows
    stems = [s for s in STEMS if s in rows and s in base_rows]
    d0, lo, hi, p, w, l = boot(base_rows, rows, stems)
    sc, adj, div = score(rows, stems)
    per = {}
    for pfx in ("44b6", "6bba"):
        ss = [s for s in stems if s.startswith(pfx)]
        per[pfx] = boot(base_rows, rows, ss, B=500) if ss else (0, 0, 0, 0, 0, 0)
    log(f"{name:<26} score={sc:.5f} d={d0:+.5f} CI[{lo:+.5f},{hi:+.5f}] P>0={p:.2f} +{w}/-{l} | forks_added={n_add} "
        f"div {div[0]}/{div[1]}/{div[2]} adj={adj:.5f} | 44b6 {per['44b6'][0]:+.5f} 6bba {per['6bba'][0]:+.5f} | {_t.time()-t1:.0f}s")
    results.append({"name": name, "cfg": cfg, "xr": xr, "score": sc, "delta": d0, "lo": lo, "hi": hi, "p": p, "win": w, "lose": l,
                    "forks_added": n_add, "div": list(div), "adj": adj, "d44": per["44b6"][0], "d6b": per["6bba"][0]})

(OUT / "results.json").write_text(_json.dumps(results, indent=1))
with (OUT / "rows.csv").open("w", newline="") as f:
    keys = ["config", "stem"] + [k for k in next(iter(ROWS["base(x138)"].values())) if k != "stem"]
    w = _csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for name, rows in ROWS.items():
        for s, r in rows.items():
            w.writerow({"config": name, **r})
log("CPU LAB 3 done")
