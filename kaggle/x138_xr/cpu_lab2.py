# ---- CPU LAB 2: score XR rule combinations with the OFFICIAL metric on every captured graph found.
import collections
import csv as _csv
import json as _json
import sys
import time as _t
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

OUT = WORKING_DIR / "cpu2"
OUT.mkdir(parents=True, exist_ok=True)
T0 = _t.time()
VOX = np.array(VOXEL_SCALE_UM, dtype=np.float64)
LAB_L7_REF = __LAB_L7_REF__


def log(*a):
    print(f"[{_t.time() - T0:6.0f}s]", *a, flush=True)


OFF = WORKING_DIR / "offmetric" / "biohub_official"
OFF.mkdir(parents=True, exist_ok=True)
for _name, _text in __OFFICIAL_FILES__.items():
    (OFF / _name).write_text(_text, encoding="utf-8")
sys.path.insert(0, str(OFF.parent))
import polars as pl
import tracksdata as td
from biohub_official.metrics import evaluate as off_evaluate, per_sample_metrics as off_per_sample, summarise as off_summarise

exec(__FILTER_SRC__, globals())
exec(__XR_SRC__, globals())

GRAPHS = {}
for p in sorted(Path("/kaggle/input").rglob("lab/graphs/*.npz")):
    if p.stem in GRAPHS:
        continue
    z = np.load(p)
    GRAPHS[p.stem] = {k: z[k] for k in z.files}
STEMS = sorted(GRAPHS)
log("graphs loaded:", len(STEMS), collections.Counter(s.split("_")[0] for s in STEMS))
TRAIN_DIR = COMP_DIR / "train"


def graph_dicts(stem):
    g = GRAPHS[stem]
    nid = g["nid"].astype(int)
    raw = g["nodes"].astype(np.float64)
    nodes = {int(i): {"node_id": int(i), "t": int(r[0]), "z": float(r[1]), "y": float(r[2]), "x": float(r[3])} for i, r in zip(nid, raw)}
    pos = {int(i): r[1:] * VOX for i, r in zip(nid, raw)}
    edges = []
    for (s, d), p in zip(g["edges"].astype(int), g["eprob"].astype(np.float64)):
        s, d = int(s), int(d)
        if s in nodes and d in nodes:
            edges.append({"source_id": s, "target_id": d, "edge_prob": (None if not np.isfinite(p) else float(p)),
                          "distance_um": float(np.linalg.norm(pos[s] - pos[d]))})
    final = {int(i): (int(r[0]), float(r[1]), float(r[2]), float(r[3])) for i, r in zip(nid, g["nodes_final"])}
    t_true = float(g["t_true"][0])
    return nodes, edges, final, (t_true if t_true > 0 else None)


CACHE = {s: graph_dicts(s) for s in STEMS}
log("dict graphs built")


def run_filter(nodes, edges, L=7):
    g = globals()
    saved = g["OUTPUT_MIN_TRACK_LEN"]
    g["OUTPUT_MIN_TRACK_LEN"] = int(L)
    try:
        return filter_short_track_components(nodes, edges, collections.Counter())
    finally:
        g["OUTPUT_MIN_TRACK_LEN"] = saved


def pipeline(stem, cfg, L=7):
    nodes, edges, _, _ = CACHE[stem]
    cfg = dict(cfg)
    L = int(cfg.pop("base_L", L))
    xr_configure(cut_nan_dist_um=0.0, cut_end_prob=0.0, cut_end_mode="both", fork_min_branch=0, prune="", cut_end_iter=1, fork_nan_only="0")
    xr_configure(**cfg)
    st = {}
    n1, e1 = xr_pre_filter(nodes, [dict(e) for e in edges], stem, st)
    n2, e2 = run_filter(n1, e1, L)
    n3, e3 = xr_post_filter(n2, e2, stem, st)
    return n3, e3, st


def build_td(plain, pairs):
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, 0.0)
    keys = sorted(plain)
    ids = graph.bulk_add_nodes([{"t": int(plain[k][0]), "z": float(plain[k][1]), "y": float(plain[k][2]), "x": float(plain[k][3])} for k in keys])
    mapping = dict(zip(keys, ids))
    if pairs:
        graph.bulk_add_edges([{"source_id": mapping[s], "target_id": mapping[d]} for s, d in pairs])
    return graph, mapping


def worker(args):
    stem, plain, pairs, t_true = args
    try:
        graph, _ = build_td(plain, pairs)
        truth = td.graph.IndexedRXGraph.from_geff(str(TRAIN_DIR / f"{stem}.geff"))[0]
        res = off_evaluate(graph, truth, scale=tuple(VOXEL_SCALE_UM), max_distance=7.0)
        row = off_per_sample(res, t_true if t_true else float("nan"), float("nan"))
        row["stem"] = stem
        return stem, row, None
    except Exception as exc:
        return stem, None, f"{type(exc).__name__}: {exc}"


def official_rows(graphs, procs=4):
    tasks = []
    for s in STEMS:
        n, e = graphs[s]
        final, t_true = CACHE[s][2], CACHE[s][3]
        tasks.append((s, {k: final[k] for k in n}, [(int(x["source_id"]), int(x["target_id"])) for x in e], t_true))
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


E_SPEC = "44b6:8:0.7,6bba:11:0.7"
COMBO = {"prune": E_SPEC, "cut_nan_dist_um": 8.0, "cut_end_prob": 0.5}
COMBO_L = {"prune": E_SPEC, "cut_nan_dist_um": 8.0, "cut_end_prob": 0.5, "cut_end_mode": "last"}
P_COMBO_L = {"prune": "*:8:0.7", "cut_nan_dist_um": 8.0, "cut_end_prob": 0.5, "cut_end_mode": "last"}
CONFIGS = [
    ("L7", {}),
    ("nan8+end5last", {"cut_nan_dist_um": 8.0, "cut_end_prob": 0.5, "cut_end_mode": "last"}),
    ("nan8+end5", {"cut_nan_dist_um": 8.0, "cut_end_prob": 0.5}),
    ("fork6-nanonly", {"fork_min_branch": 6, "fork_nan_only": "1"}),
    ("fork7-nanonly", {"fork_min_branch": 7, "fork_nan_only": "1"}),
    ("fork8-nanonly", {"fork_min_branch": 8, "fork_nan_only": "1"}),
    ("fork9-nanonly", {"fork_min_branch": 9, "fork_nan_only": "1"}),
    ("E+fork8-nanonly", {"prune": E_SPEC, "fork_min_branch": 8, "fork_nan_only": "1"}),
    ("P8+nan8+end5last", dict(P_COMBO_L)),
    ("P8+nan8+end5last+fork8-nanonly", dict(P_COMBO_L, fork_min_branch=8, fork_nan_only="1")),
    ("E+nan8+end5last", dict(COMBO_L)),
    ("E+nan8+end5last+fork7-nanonly", dict(COMBO_L, fork_min_branch=7, fork_nan_only="1")),
    ("E+nan8+end5last+fork8-nanonly", dict(COMBO_L, fork_min_branch=8, fork_nan_only="1")),
    ("E+nan8+end5+fork8-nanonly", dict(COMBO, fork_min_branch=8, fork_nan_only="1")),
    ("E+nan8+end6last+fork8-nanonly", dict(COMBO_L, cut_end_prob=0.6, fork_min_branch=8, fork_nan_only="1")),
]

# ---- fork table with OFFICIAL labels on the L7 graph: cumulative TP/FP forks by shorter-branch length
from biohub_official import division_metrics as offdiv


def fork_worker(args):
    stem, plain, pairs, t_true = args
    try:
        graph, mapping = build_td(plain, pairs)
        inv = {v: k for k, v in mapping.items()}
        truth = td.graph.IndexedRXGraph.from_geff(str(TRAIN_DIR / f"{stem}.geff"))[0]
        ds = offdiv.score_divisions(graph, truth, scale=tuple(VOXEL_SCALE_UM), max_distance=7.0)
        return stem, ([inv[i] for i in ds.tp_forks], [inv[i] for i in ds.fp_forks]), None
    except Exception as exc:
        return stem, None, f"{type(exc).__name__}: {exc}"


try:
    tasks = []
    L7G = {}
    for s in STEMS:
        n, e, _ = pipeline(s, {})
        L7G[s] = (n, e)
        final, t_true = CACHE[s][2], CACHE[s][3]
        tasks.append((s, {k: final[k] for k in n}, [(int(x["source_id"]), int(x["target_id"])) for x in e], t_true))
    labels = {}
    with Pool(4) as pool:
        for stem, lab, err in pool.imap_unordered(fork_worker, tasks):
            if err:
                log("fork label error", stem, err)
            else:
                labels[stem] = lab
    hist = {"TP": collections.Counter(), "FP": collections.Counter(), "all": collections.Counter()}
    hist_pfx = {p: {"TP": collections.Counter(), "FP": collections.Counter()} for p in ("44b6", "6bba")}
    for s, (tp_f, fp_f) in labels.items():
        n, e = L7G[s]
        succ = collections.defaultdict(list); pred = collections.defaultdict(list); prob = {}
        for x in e:
            a, b = int(x["source_id"]), int(x["target_id"])
            succ[a].append(b); pred[b].append(a); prob[(a, b)] = x["edge_prob"]

        def down(nid, cap=64):
            k = 1
            while k < cap:
                nx = succ.get(nid, [])
                if len(nx) != 1 or len(pred.get(nx[0], [])) != 1:
                    return k
                nid = nx[0]; k += 1
            return k
        tp_set, fp_set = set(tp_f), set(fp_f)
        for p_, ch in succ.items():
            if len(ch) < 2:
                continue
            dm = min(down(c) for c in ch)
            hist["all"][dm] += 1
            lab = "TP" if p_ in tp_set else "FP" if p_ in fp_set else None
            if lab:
                hist[lab][dm] += 1
                hist_pfx[s.split("_")[0]][lab][dm] += 1
    log(f"FORK TABLE (official labels, {len(labels)} videos): TP forks={sum(hist['TP'].values())} FP forks={sum(hist['FP'].values())} all forks={sum(hist['all'].values())}")
    log("  shorter-branch length L: cumulative forks with length < L  [all | TP | FP]  (44b6 TP/FP, 6bba TP/FP)")
    for L in range(2, 26):
        ca = sum(v for k, v in hist["all"].items() if k < L); ct = sum(v for k, v in hist["TP"].items() if k < L); cf = sum(v for k, v in hist["FP"].items() if k < L)
        p44 = (sum(v for k, v in hist_pfx["44b6"]["TP"].items() if k < L), sum(v for k, v in hist_pfx["44b6"]["FP"].items() if k < L))
        p6b = (sum(v for k, v in hist_pfx["6bba"]["TP"].items() if k < L), sum(v for k, v in hist_pfx["6bba"]["FP"].items() if k < L))
        log(f"    L={L:2d}: all<{L}={ca:5d}  TP<{L}={ct:3d}  FP<{L}={cf:3d}   44b6 {p44[0]}/{p44[1]}  6bba {p6b[0]}/{p6b[1]}")
    log("  TP fork shorter-branch lengths (sorted):", sorted(k for k, v in hist["TP"].items() for _ in range(v))[:60])
except Exception:
    traceback.print_exc()

ROWS = {}
STATS = {}
results = []
base = None
for name, cfg in CONFIGS:
    t1 = _t.time()
    graphs, st = {}, collections.Counter()
    for s in STEMS:
        n, e, sst = pipeline(s, cfg)
        graphs[s] = (n, e)
        for k, v in sst.items():
            st[k] += v
    rows = official_rows(graphs)
    ROWS[name] = rows
    STATS[name] = dict(st)
    if base is None:
        base = rows
        mism = [s for s in STEMS if s in LAB_L7_REF and abs(rows[s]["adj_edge_jaccard"] - LAB_L7_REF[s]) > 1e-6]
        log("SELF-CHECK official L7 adj vs GPU lab:", len(mism), "mismatches")
    stems = [s for s in STEMS if s in rows and s in base]
    d0, lo, hi, p, w, l = boot(base, rows, stems)
    sc, adj, div = score(rows, stems)
    per = {}
    for pfx in ("44b6", "6bba"):
        ss = [s for s in stems if s.startswith(pfx)]
        per[pfx] = boot(base, rows, ss, B=600) if ss else (0, 0, 0, 0, 0, 0)
    tpred = sum(rows[s]["num_pred_nodes"] for s in stems)
    log(f"{name:<24} score={sc:.5f} d={d0:+.5f} CI[{lo:+.5f},{hi:+.5f}] P>0={p:.2f} +{w}/-{l} | "
        f"44b6 {per['44b6'][0]:+.5f}(P{per['44b6'][3]:.2f}) 6bba {per['6bba'][0]:+.5f}(P{per['6bba'][3]:.2f}) | "
        f"adj={adj:.5f} div {div[0]}/{div[1]}/{div[2]} Tpred={tpred} stats={dict(st)} | {_t.time()-t1:.0f}s")
    results.append({"name": name, "cfg": cfg, "score": sc, "delta": d0, "lo": lo, "hi": hi, "p": p, "win": w, "lose": l,
                    "d44": per["44b6"][0], "p44": per["44b6"][3], "d6b": per["6bba"][0], "p6b": per["6bba"][3],
                    "adj": adj, "div": list(div), "tpred": tpred, "stats": dict(st)})

(OUT / "results.json").write_text(_json.dumps(results, indent=1))
with (OUT / "rows.csv").open("w", newline="") as f:
    keys = ["config", "stem"] + [k for k in next(iter(ROWS["L7"].values())) if k != "stem"]
    w = _csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for name, rows in ROWS.items():
        for s, r in rows.items():
            w.writerow({"config": name, **r})
log("CPU LAB 2 done")
