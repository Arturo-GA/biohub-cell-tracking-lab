# ---- CPU LAB: offline analysis of the 72 captured pre-filter graphs (lab0 + lab1 outputs).
# Sections are independent (each in try/except) and write to /kaggle/working/cpu/.
import collections
import csv as _csv
import json as _json
import sys
import time as _t
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

OUT = WORKING_DIR / "cpu"
OUT.mkdir(parents=True, exist_ok=True)
T0 = _t.time()
VOX = np.array(VOXEL_SCALE_UM, dtype=np.float64)
VALIDATOR_MATCH_RADIUS_UM = 7.0
VALIDATOR_NODE_COUNT_PENALTY_A = 0.1
VALIDATOR_DIVISION_WEIGHT = 0.1
LAB_L7_REF = __LAB_L7_REF__   # stem -> adjusted_edge_jaccard of rule L7 in the GPU lab (self-check)


def log(*a):
    print(f"[{_t.time() - T0:6.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- official metric package
OFF = WORKING_DIR / "offmetric" / "biohub_official"
OFF.mkdir(parents=True, exist_ok=True)
for _name, _text in __OFFICIAL_FILES__.items():
    (OFF / _name).write_text(_text, encoding="utf-8")
sys.path.insert(0, str(OFF.parent))
import polars as pl
import tracksdata as td
from biohub_official.metrics import evaluate as off_evaluate, per_sample_metrics as off_per_sample, summarise as off_summarise
from biohub_official import division_metrics as offdiv

# ---------------------------------------------------------------- x138 short-track filter (exact source)
exec(__FILTER_SRC__, globals())
assert "filter_short_track_components" in globals()

# ---------------------------------------------------------------- load captured graphs
GRAPHS = {}
for p in sorted(Path("/kaggle/input").rglob("lab/graphs/*.npz")):
    z = np.load(p)
    GRAPHS[p.stem] = {k: z[k] for k in z.files}
log("graphs loaded:", len(GRAPHS))
STEMS = sorted(GRAPHS)
TRAIN_DIR = COMP_DIR / "train"


def prefix(stem):
    return stem.split("_")[0]


def graph_dicts(stem):
    """Pre-filter graph as the dict structures x138's filter expects (raw coords, edge_prob None for NaN)."""
    g = GRAPHS[stem]
    nid = g["nid"].astype(int)
    raw = g["nodes"].astype(np.float64)
    nodes = {int(i): {"node_id": int(i), "t": int(r[0]), "z": float(r[1]), "y": float(r[2]), "x": float(r[3])}
             for i, r in zip(nid, raw)}
    pos = {int(i): r[1:] * VOX for i, r in zip(nid, raw)}
    edges = []
    for (s, d), p in zip(g["edges"].astype(int), g["eprob"].astype(np.float64)):
        s, d = int(s), int(d)
        if s not in nodes or d not in nodes:
            continue
        edges.append({"source_id": s, "target_id": d, "edge_prob": (None if not np.isfinite(p) else float(p)),
                      "distance_um": float(np.linalg.norm(pos[s] - pos[d]))})
    final = {int(i): (int(r[0]), float(r[1]), float(r[2]), float(r[3])) for i, r in zip(nid, g["nodes_final"])}
    gt_nodes = {int(i): (int(r[0]), float(r[1]), float(r[2]), float(r[3])) for i, r in zip(g["gid"].astype(int), g["gnodes"])}
    gt_edges = [(int(a), int(b)) for a, b in g["gedges"].astype(int)]
    t_true = float(g["t_true"][0])
    return nodes, edges, final, gt_nodes, gt_edges, (t_true if t_true > 0 else None)


CACHE = {s: graph_dicts(s) for s in STEMS}
log("dict graphs built")


def run_filter(nodes, edges, L):
    g = globals()
    saved = g["OUTPUT_MIN_TRACK_LEN"]
    g["OUTPUT_MIN_TRACK_LEN"] = int(L)
    try:
        return filter_short_track_components(nodes, edges, collections.Counter())
    finally:
        g["OUTPUT_MIN_TRACK_LEN"] = saved


def score_graph(stem, nodes, edges):
    _, _, final, gt_nodes, gt_edges, t_true = CACHE[stem]
    plain = {k: final[k] for k in nodes}
    pe = [(int(e["source_id"]), int(e["target_id"])) for e in edges]
    row = score_sample(plain, pe, gt_nodes, gt_edges, t_true)
    row["stem"] = stem
    return row


def agg(rows):
    return aggregate_official(rows)


def boot_delta(rows_a, rows_b, B=1500, seed=0):
    """Paired stem bootstrap of proxy(b) - proxy(a)."""
    A = {r["stem"]: r for r in rows_a}; Bm = {r["stem"]: r for r in rows_b}
    stems = sorted(set(A) & set(Bm))
    rng = np.random.RandomState(seed)
    d0 = agg([Bm[s] for s in stems])["proxy_score"] - agg([A[s] for s in stems])["proxy_score"]
    ds = []
    for _ in range(B):
        samp = [stems[i] for i in rng.randint(0, len(stems), len(stems))]
        ds.append(agg([Bm[s] for s in samp])["proxy_score"] - agg([A[s] for s in samp])["proxy_score"])
    ds = np.sort(ds)
    win = sum(1 for s in stems if agg([Bm[s]])["proxy_score"] > agg([A[s]])["proxy_score"] + 1e-12)
    lose = sum(1 for s in stems if agg([Bm[s]])["proxy_score"] < agg([A[s]])["proxy_score"] - 1e-12)
    return d0, ds[int(0.05 * B)], ds[int(0.95 * B)], float((ds > 0).mean()), win, lose


def report(label, rows, base_rows):
    d0, lo, hi, p, w, l = boot_delta(base_rows, rows)
    by = {}
    for pfx in ("44b6", "6bba"):
        d = boot_delta([r for r in base_rows if r["stem"].startswith(pfx)], [r for r in rows if r["stem"].startswith(pfx)], B=600)
        by[pfx] = d
    s = agg(rows)
    line = (f"{label:<34} proxy={s['proxy_score']:.5f} d={d0:+.5f} CI[{lo:+.5f},{hi:+.5f}] P>0={p:.2f} +{w}/-{l} | "
            f"44b6 {by['44b6'][0]:+.5f} (P{by['44b6'][3]:.2f}) 6bba {by['6bba'][0]:+.5f} (P{by['6bba'][3]:.2f}) | "
            f"div {s['div_tp']}/{s['div_fp']}/{s['div_fn']} Tpred={int(sum(r['t_pred'] for r in rows))}")
    log(line)
    return {"label": label, "proxy": s["proxy_score"], "delta": d0, "lo": lo, "hi": hi, "p": p, "win": w, "lose": l,
            "d44": by["44b6"][0], "p44": by["44b6"][3], "d6b": by["6bba"][0], "p6b": by["6bba"][3],
            "adj": s["adjusted_edge_jaccard"], "divJ": s["division_jaccard"]}


RESULTS = []
ALL_ROWS = {}

# ================================================================ 0. baseline L7 + self-check
L7 = {}
base_rows = []
for s in STEMS:
    nodes, edges, *_ = CACHE[s]
    n7, e7 = run_filter(nodes, edges, 7)
    L7[s] = (n7, e7)
    base_rows.append(score_graph(s, n7, e7))
ALL_ROWS["L7"] = base_rows
mism = [(r["stem"], r["adjusted_edge_jaccard"], LAB_L7_REF.get(r["stem"])) for r in base_rows
        if r["stem"] in LAB_L7_REF and abs(r["adjusted_edge_jaccard"] - LAB_L7_REF[r["stem"]]) > 1e-6]
log("SELF-CHECK L7 vs GPU lab: mismatches =", len(mism), mism[:3])
RESULTS.append(report("L7 (baseline)", base_rows, base_rows))

# ================================================================ 1. edge-level FP/TP table on the L7 graph
def edge_table(stem):
    nodes, edges = L7[stem]
    _, _, final, gt_nodes, gt_edges, _ = CACHE[stem]
    plain = {k: final[k] for k in nodes}
    p2g, _ = match_nodes_bipartite(plain, gt_nodes, max_dist=VALIDATOR_MATCH_RADIUS_UM)
    gt_out = collections.defaultdict(set); gt_in = {}
    for a, b in gt_edges:
        gt_out[a].add(b); gt_in[b] = a
    succ = collections.defaultdict(list); pred = collections.defaultdict(list)
    for e in edges:
        succ[int(e["source_id"])].append(int(e["target_id"])); pred[int(e["target_id"])].append(int(e["source_id"]))
    din = {e["source_id"]: 0 for e in edges}
    dist_in = {}; dist_out = collections.defaultdict(list)
    for e in edges:
        dist_in[int(e["target_id"])] = e["distance_um"]; dist_out[int(e["source_id"])].append(e["distance_um"])
    # component sizes
    parent = {n: n for n in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for e in edges:
        a, b = find(int(e["source_id"])), find(int(e["target_id"]))
        if a != b: parent[a] = b
    csize = collections.Counter(find(n) for n in nodes)
    rows = []
    for e in edges:
        s, d = int(e["source_id"]), int(e["target_id"])
        ms, md = p2g.get(s), p2g.get(d)
        tp = ms is not None and md is not None and md in gt_out.get(ms, ())
        fp = (not tp) and ((md is not None and md in gt_in) or (ms is not None and bool(gt_out.get(ms))))
        nb = [dist_in[s]] if s in dist_in else []
        nb += [x for x in dist_out.get(d, [])]
        ref = float(np.median(nb)) if nb else float("nan")
        rows.append({
            "stem": stem, "pfx": prefix(stem), "s": s, "d": d, "t": nodes[s]["t"],
            "prob": (-1.0 if e["edge_prob"] is None else e["edge_prob"]), "nan": int(e["edge_prob"] is None),
            "dist": e["distance_um"], "jump": (e["distance_um"] / max(ref, 0.5) if ref == ref else -1.0),
            "first": int(not pred.get(s)), "last": int(not succ.get(d)),
            "fork": int(len(succ.get(s, [])) >= 2), "csize": csize[find(s)],
            "label": ("TP" if tp else "FP" if fp else "-"),
        })
    return rows

try:
    ET = []
    for s in STEMS:
        ET.extend(edge_table(s))
    lab = [r for r in ET if r["label"] != "-"]
    log(f"edges total={len(ET)} labeled TP={sum(r['label']=='TP' for r in lab)} FP={sum(r['label']=='FP' for r in lab)}")
    with (OUT / "edges_labeled.csv").open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(lab[0])); w.writeheader(); w.writerows(lab)

    def bucket_report(name, fn):
        tot = collections.Counter(); tp = collections.Counter(); fp = collections.Counter()
        for r in ET:
            b = fn(r)
            tot[b] += 1
            if r["label"] == "TP": tp[b] += 1
            elif r["label"] == "FP": fp[b] += 1
        log(f"--- edge buckets by {name}: bucket n TP FP FPrate(FP/(TP+FP)) cut_gain(FP*0.9-TP)")
        for b in sorted(tot):
            t_, f_ = tp[b], fp[b]
            rate = f_ / max(t_ + f_, 1)
            log(f"    {str(b):<18} n={tot[b]:7d} TP={t_:6d} FP={f_:5d} rate={rate:.3f} gain={0.9*f_ - t_:+.0f}")
    def pb(p):
        if p < 0: return "relink(NaN)"
        for e in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
            if p < e: return f"p<{e}"
        return "p>=0.95"
    bucket_report("prob", lambda r: pb(r["prob"]))
    bucket_report("prob x pfx", lambda r: (r["pfx"], pb(r["prob"])))
    bucket_report("dist(um)", lambda r: f"d<{e}" if False else next((f"d<{e}" for e in (2, 4, 6, 8, 10, 12) if r["dist"] < e), "d>=12"))
    bucket_report("NaN x dist", lambda r: (r["nan"], next((f"d<{e}" for e in (4, 6, 8, 10) if r["dist"] < e), "d>=10")))
    bucket_report("jump ratio", lambda r: "n/a" if r["jump"] < 0 else next((f"j<{e}" for e in (1.5, 2, 2.5, 3, 4) if r["jump"] < e), "j>=4"))
    bucket_report("position", lambda r: ("first" if r["first"] else "") + ("last" if r["last"] else "") or "mid")
    bucket_report("position x prob<0.5", lambda r: (("first" if r["first"] else "") + ("last" if r["last"] else "") or "mid", r["prob"] < 0.5))
    bucket_report("fork edge x prob", lambda r: (r["fork"], pb(r["prob"])))
    bucket_report("csize", lambda r: next((f"c<{e}" for e in (8, 10, 15, 25, 50) if r["csize"] < e), "c>=50"))
except Exception:
    traceback.print_exc()

# ================================================================ 2. edge-cut rules (cut on the pre-filter graph, then L7 filter)
def apply_cut(stem, keep_fn):
    nodes, edges = CACHE[stem][0], CACHE[stem][1]
    succ = collections.defaultdict(list); pred = collections.defaultdict(list)
    for e in edges:
        succ[int(e["source_id"])].append(e); pred[int(e["target_id"])].append(e)
    ctx = {"succ": succ, "pred": pred, "nodes": nodes}
    kept = [e for e in edges if keep_fn(e, ctx)]
    return run_filter(nodes, kept, 7)

CUT_RULES = {
    "cut NaN d>8": lambda e, c: not (e["edge_prob"] is None and e["distance_um"] > 8),
    "cut NaN d>10": lambda e, c: not (e["edge_prob"] is None and e["distance_um"] > 10),
    "cut NaN d>12": lambda e, c: not (e["edge_prob"] is None and e["distance_um"] > 12),
    "cut p<0.3": lambda e, c: not (e["edge_prob"] is not None and e["edge_prob"] < 0.3),
    "cut p<0.5": lambda e, c: not (e["edge_prob"] is not None and e["edge_prob"] < 0.5),
    "cut any d>10": lambda e, c: not (e["distance_um"] > 10),
    "cut any d>12": lambda e, c: not (e["distance_um"] > 12),
    "cut last-edge p<0.5": lambda e, c: not (e["edge_prob"] is not None and e["edge_prob"] < 0.5 and not c["succ"].get(int(e["target_id"]))),
    "cut first-edge p<0.5": lambda e, c: not (e["edge_prob"] is not None and e["edge_prob"] < 0.5 and not c["pred"].get(int(e["source_id"]))),
    "cut NaN last-edge": lambda e, c: not (e["edge_prob"] is None and not c["succ"].get(int(e["target_id"]))),
}
try:
    for label, fn in CUT_RULES.items():
        rows = []
        for s in STEMS:
            n2, e2 = apply_cut(s, fn)
            rows.append(score_graph(s, n2, e2))
        ALL_ROWS[label] = rows
        RESULTS.append(report(label, rows, base_rows))
except Exception:
    traceback.print_exc()

# ================================================================ 3. GT annotation density map (GT share / pred share)
try:
    def hist_share(vals_gt, vals_pr, edges):
        hg, _ = np.histogram(vals_gt, bins=edges); hp, _ = np.histogram(vals_pr, bins=edges)
        return hg / max(hg.sum(), 1), hp / max(hp.sum(), 1), hg, hp
    for pfx in ("44b6", "6bba", "all"):
        zg, zp, tg, tpp, mg, mp = [], [], [], [], [], []
        for s in STEMS:
            if pfx != "all" and not s.startswith(pfx): continue
            nodes, edges = L7[s]
            _, _, final, gt_nodes, _, _ = CACHE[s]
            P = np.array([final[k][1:] for k in nodes]); G = np.array([v[1:] for v in gt_nodes.values()])
            Tp = np.array([final[k][0] for k in nodes]); Tg = np.array([v[0] for v in gt_nodes.values()])
            zmax = max(P[:, 0].max(), G[:, 0].max()); ymax = max(P[:, 1].max(), G[:, 1].max()); xmax = max(P[:, 2].max(), G[:, 2].max())
            tmax = max(Tp.max(), Tg.max())
            zg.append(G[:, 0] / max(zmax, 1)); zp.append(P[:, 0] / max(zmax, 1))
            tg.append(Tg / max(tmax, 1)); tpp.append(Tp / max(tmax, 1))
            # xy margin in voxels (distance to nearest lateral border)
            mg.append(np.minimum.reduce([G[:, 1], ymax - G[:, 1], G[:, 2], xmax - G[:, 2]]))
            mp.append(np.minimum.reduce([P[:, 1], ymax - P[:, 1], P[:, 2], xmax - P[:, 2]]))
        for name, a, b, edges_ in [("z_rel", zg, zp, np.linspace(0, 1.0001, 11)), ("t_rel", tg, tpp, np.linspace(0, 1.0001, 11)),
                                    ("xy_margin_vox", mg, mp, np.array([0, 4, 8, 16, 32, 64, 128, 10000]))]:
            sg, sp, hg, hp = hist_share(np.concatenate(a), np.concatenate(b), edges_)
            log(f"--- GT density map {pfx} by {name}: bin | pred share | GT share | ratio GT/pred | pred nodes")
            for i in range(len(hg)):
                log(f"    [{edges_[i]:7.2f},{edges_[i+1]:7.2f}) pred={sp[i]:.4f} gt={sg[i]:.4f} ratio={sg[i]/max(sp[i],1e-9):.2f} n_pred={hp[i]}")
except Exception:
    traceback.print_exc()

# ================================================================ 4. per-embryo size x prob grid (after L7)
def comp_index(nodes, edges):
    parent = {n: n for n in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    out = collections.Counter(); probs = collections.defaultdict(list)
    for e in edges:
        s, d = int(e["source_id"]), int(e["target_id"])
        out[s] += 1
        a, b = find(s), find(d)
        if a != b: parent[a] = b
    comps = collections.defaultdict(list)
    for n in nodes: comps[find(n)].append(n)
    for e in edges:
        if e["edge_prob"] is not None: probs[find(int(e["source_id"]))].append(e["edge_prob"])
    return comps, out, probs

def prune_size_prob(nodes, edges, S, P):
    comps, out, probs = comp_index(nodes, edges)
    rm = set()
    for r, ms in comps.items():
        if len(ms) >= S or any(out.get(m, 0) >= 2 for m in ms): continue
        mp = float(np.mean(probs[r])) if probs[r] else 0.0
        if mp < P: rm.update(ms)
    if not rm: return nodes, edges
    kn = {k: v for k, v in nodes.items() if k not in rm}
    return kn, [e for e in edges if int(e["source_id"]) in kn and int(e["target_id"]) in kn]

try:
    GRID = {}
    for S in (8, 9, 10, 11, 12):
        for P in (0.5, 0.7, 0.8, 0.85, 0.9):
            rows = []
            for s in STEMS:
                n7, e7 = L7[s]
                n2, e2 = prune_size_prob(n7, e7, S, P)
                rows.append(score_graph(s, n2, e2))
            GRID[(S, P)] = rows
            ALL_ROWS[f"L7+prob{P}<{S}"] = rows
            RESULTS.append(report(f"L7+prob{P}<{S}", rows, base_rows))
    # best per embryo and the combined embryo-specific rule
    best = {}
    for pfx in ("44b6", "6bba"):
        cands = []
        for (S, P), rows in GRID.items():
            d = boot_delta([r for r in base_rows if r["stem"].startswith(pfx)], [r for r in rows if r["stem"].startswith(pfx)], B=400)
            cands.append((d[0], d[3], S, P))
        cands.sort(reverse=True)
        best[pfx] = cands[0]
        log(f"best {pfx}: " + ", ".join(f"S{S}P{P}:{d:+.5f}(P{p:.2f})" for d, p, S, P in cands[:5]))
    rows = []
    for s in STEMS:
        _, _, S, P = best[prefix(s)]
        rows.append(GRID[(S, P)][STEMS.index(s)])
    ALL_ROWS["embryo-specific best"] = rows
    RESULTS.append(report(f"embryo-specific {best['44b6'][2:]},{best['6bba'][2:]}", rows, base_rows))
except Exception:
    traceback.print_exc()

# ================================================================ 5. OFFICIAL metric + fork labels
def build_td(nodes_final_plain, edge_pairs):
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, 0.0)
    keys = sorted(nodes_final_plain)
    ids = graph.bulk_add_nodes([{"t": int(nodes_final_plain[k][0]), "z": float(nodes_final_plain[k][1]),
                                 "y": float(nodes_final_plain[k][2]), "x": float(nodes_final_plain[k][3])} for k in keys])
    mapping = dict(zip(keys, ids))
    if edge_pairs:
        graph.bulk_add_edges([{"source_id": mapping[s], "target_id": mapping[d]} for s, d in edge_pairs])
    return graph, mapping

def official_worker(args):
    stem, plain, pairs, t_true, want_forks = args
    try:
        graph, mapping = build_td(plain, pairs)
        truth = td.graph.IndexedRXGraph.from_geff(str(TRAIN_DIR / f"{stem}.geff"))[0]
        res = off_evaluate(graph, truth, scale=tuple(VOXEL_SCALE_UM), max_distance=7.0)
        row = off_per_sample(res, t_true if t_true else float("nan"), float("nan"))
        row["stem"] = stem
        forks = None
        if want_forks:
            inv = {v: k for k, v in mapping.items()}
            ds = offdiv.score_divisions(graph, truth, scale=tuple(VOXEL_SCALE_UM), max_distance=7.0)
            ev, cross, malformed = offdiv._pred_division_fork_sets(graph, truth, tuple(VOXEL_SCALE_UM), 7.0)
            forks = {"tp": [inv[i] for i in ds.tp_forks], "fp": [inv[i] for i in ds.fp_forks],
                     "evaluable": [inv[i] for i in ev], "cross": [inv[i] for i in cross], "malformed": [inv[i] for i in malformed]}
        return stem, row, forks, None
    except Exception as exc:
        return stem, None, None, f"{type(exc).__name__}: {exc}"

def official_eval(graphs_by_stem, want_forks=False, procs=4):
    tasks = []
    for s in STEMS:
        nodes, edges = graphs_by_stem[s]
        final, t_true = CACHE[s][2], CACHE[s][5]
        plain = {k: final[k] for k in nodes}
        pairs = [(int(e["source_id"]), int(e["target_id"])) for e in edges]
        tasks.append((s, plain, pairs, t_true, want_forks))
    rows, forks, errs = {}, {}, {}
    with Pool(procs) as pool:
        for stem, row, fk, err in pool.imap_unordered(official_worker, tasks):
            if err: errs[stem] = err
            else:
                rows[stem] = row; forks[stem] = fk
    if errs: log("official errors:", list(errs.items())[:3])
    return rows, forks

def off_summary(rows):
    s = off_summarise([rows[k] for k in sorted(rows)])
    return s

OFF_ROWS = {}
try:
    t1 = _t.time()
    off_l7, FORKS = official_eval(L7, want_forks=True)
    OFF_ROWS["L7"] = off_l7
    s = off_summary(off_l7)
    log(f"OFFICIAL L7: score={s['score']:.5f} adj={s['adj_edge_jaccard']:.5f} divJ={s['division_jaccard']:.4f} "
        f"div {s['division_tp']}/{s['division_fp']}/{s['division_fn']} | cell8 proxy={agg(base_rows)['proxy_score']:.5f} | {_t.time()-t1:.0f}s")
    for pfx in ("44b6", "6bba"):
        sp = off_summarise([off_l7[k] for k in sorted(off_l7) if k.startswith(pfx)])
        log(f"   {pfx}: score={sp['score']:.5f} adj={sp['adj_edge_jaccard']:.5f} div {sp['division_tp']}/{sp['division_fp']}/{sp['division_fn']}")
    # per-stem comparison of cell8 vs official adj
    diffs = [(r["stem"], r["adjusted_edge_jaccard"] - off_l7[r["stem"]]["adj_edge_jaccard"]) for r in base_rows if r["stem"] in off_l7]
    log("cell8-official adj per stem: mean=%.5f max|d|=%.5f" % (np.mean([d for _, d in diffs]), max(abs(d) for _, d in diffs)))
except Exception:
    traceback.print_exc()
    FORKS = {}

# ---------------------------------------------------------------- fork feature table
def fork_table(stem):
    nodes, edges = L7[stem]
    nodes_pre, edges_pre = CACHE[stem][0], CACHE[stem][1]
    final = CACHE[stem][2]
    succ = collections.defaultdict(list); pred = collections.defaultdict(list); eprob = {}
    for e in edges:
        s, d = int(e["source_id"]), int(e["target_id"])
        succ[s].append(d); pred[d].append(s); eprob[(s, d)] = e["edge_prob"]
    # track ends in the PRE-filter graph per frame (for "steal" detection)
    succ_pre = collections.defaultdict(list)
    for e in edges_pre: succ_pre[int(e["source_id"])].append(int(e["target_id"]))
    ends_by_t = collections.defaultdict(list)
    for n, v in nodes_pre.items():
        if not succ_pre.get(n): ends_by_t[v["t"]].append(n)
    pos = lambda n: np.array([nodes_pre[n]["z"], nodes_pre[n]["y"], nodes_pre[n]["x"]]) * VOX
    trees = {t: (cKDTree(np.array([pos(n) for n in ns])), ns) for t, ns in ends_by_t.items() if ns}
    def chain_len_down(n, cap=60):
        k = 0
        while k < cap:
            nxt = succ.get(n, [])
            if len(nxt) != 1 or len(pred.get(nxt[0], [])) != 1:
                return k + 1
            n = nxt[0]; k += 1
        return k + 1
    def chain_len_up(n, cap=60):
        k = 0
        while k < cap:
            prv = pred.get(n, [])
            if len(prv) != 1 or len(succ.get(prv[0], [])) != 1:
                return k + 1
            n = prv[0]; k += 1
        return k + 1
    fk = FORKS.get(stem) or {}
    lab = {}
    for n in fk.get("tp", []): lab[n] = "TP"
    for n in fk.get("fp", []): lab.setdefault(n, "FP")
    cls = {}
    for n in fk.get("cross", []): cls[n] = "cross"
    for n in fk.get("malformed", []): cls[n] = "malformed"
    for n in fk.get("evaluable", []): cls.setdefault(n, "evaluable")
    rows = []
    for p_, ch in succ.items():
        if len(ch) < 2: continue
        t = nodes[p_]["t"]
        c1, c2 = ch[0], ch[1]
        pr = [eprob[(p_, c)] for c in (c1, c2)]
        d1, d2 = [float(np.linalg.norm(pos(p_) - pos(c))) for c in (c1, c2)]
        sis = float(np.linalg.norm(pos(c1) - pos(c2)))
        # divergence at t+2
        g1 = succ.get(c1, []); g2 = succ.get(c2, [])
        div2 = float(np.linalg.norm(pos(g1[0]) - pos(g2[0]))) - sis if g1 and g2 else float("nan")
        # steal evidence: nearest pre-filter track end at time t (excluding parent) to each child
        steal = []
        for c in (c1, c2):
            tr = trees.get(t)
            if tr is None: steal.append(float("inf")); continue
            dists, idxs = tr[0].query(pos(c), k=min(3, len(tr[1])))
            dists = np.atleast_1d(dists); idxs = np.atleast_1d(idxs)
            best = float("inf")
            for dd, ii in zip(dists, idxs):
                if tr[1][ii] != p_: best = float(dd); break
            steal.append(best)
        rows.append({
            "stem": stem, "pfx": prefix(stem), "parent": p_, "t": t, "label": lab.get(p_, "-"), "cls": cls.get(p_, "-"),
            "p1": -1 if pr[0] is None else pr[0], "p2": -1 if pr[1] is None else pr[1], "nan_any": int(pr[0] is None or pr[1] is None),
            "pmin": min([x for x in pr if x is not None], default=-1), "d1": d1, "d2": d2, "sister": sis, "diverge2": div2,
            "up": chain_len_up(p_), "down1": chain_len_down(c1), "down2": chain_len_down(c2),
            "down_min": min(chain_len_down(c1), chain_len_down(c2)), "steal_min": min(steal), "steal_c1": steal[0], "steal_c2": steal[1],
            "c_short": c1 if chain_len_down(c1) <= chain_len_down(c2) else c2,
            "c_steal": c1 if steal[0] <= steal[1] else c2,
        })
    return rows

try:
    FT = []
    for s in STEMS: FT.extend(fork_table(s))
    with (OUT / "forks.csv").open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(FT[0])); w.writeheader(); w.writerows(FT)
    log(f"forks total={len(FT)} TP={sum(r['label']=='TP' for r in FT)} FP={sum(r['label']=='FP' for r in FT)} | "
        f"classes: {collections.Counter(r['cls'] for r in FT if r['label']=='FP')}")
    def fstat(name, fn):
        tot = collections.Counter(); tp = collections.Counter(); fp = collections.Counter()
        for r in FT:
            b = fn(r); tot[b] += 1
            if r["label"] == "TP": tp[b] += 1
            elif r["label"] == "FP": fp[b] += 1
        log(f"--- forks by {name}: bucket n TP FP")
        for b in sorted(tot, key=str):
            log(f"    {str(b):<22} n={tot[b]:5d} TP={tp[b]:3d} FP={fp[b]:3d}")
    fstat("pfx", lambda r: r["pfx"])
    fstat("nan_any", lambda r: r["nan_any"])
    fstat("pmin", lambda r: "nan-only" if r["pmin"] < 0 else next((f"<{e}" for e in (0.3, 0.5, 0.7, 0.9) if r["pmin"] < e), ">=0.9"))
    fstat("down_min", lambda r: next((f"<{e}" for e in (2, 3, 4, 6, 10) if r["down_min"] < e), ">=10"))
    fstat("up", lambda r: next((f"<{e}" for e in (2, 3, 5, 10) if r["up"] < e), ">=10"))
    fstat("sister um", lambda r: next((f"<{e}" for e in (4, 6, 8, 10, 12) if r["sister"] < e), ">=12"))
    fstat("steal_min um", lambda r: next((f"<{e}" for e in (2, 3, 4, 6, 8) if r["steal_min"] < e), ">=8"))
    fstat("diverge2", lambda r: "nan" if r["diverge2"] != r["diverge2"] else next((f"<{e}" for e in (0, 1, 2, 4) if r["diverge2"] < e), ">=4"))
    fstat("t_rel", lambda r: next((f"<{e}" for e in (10, 30, 60, 90) if r["t"] < e), ">=90"))
except Exception:
    traceback.print_exc()
    FT = []

# ---------------------------------------------------------------- division repair rules, scored OFFICIALLY
def repaired_graph(stem, rule):
    """Apply a fork rule on the PRE-filter graph then L7 filter. rule(row, ctx) -> ('drop', child) | ('steal', child, new_parent) | None"""
    nodes, edges = CACHE[stem][0], CACHE[stem][1]
    succ = collections.defaultdict(list); pred = collections.defaultdict(list); eprob = {}
    for e in edges:
        s, d = int(e["source_id"]), int(e["target_id"])
        succ[s].append(d); pred[d].append(s); eprob[(s, d)] = e["edge_prob"]
    ft = {r["parent"]: r for r in FT if r["stem"] == stem}
    drop = set(); add = []
    for p_, r in ft.items():
        if len(succ.get(p_, [])) < 2: continue
        act = rule(r, {"succ": succ, "pred": pred})
        if not act: continue
        if act[0] == "drop":
            drop.add((p_, act[1]))
        elif act[0] == "steal":
            c, newp = act[1], act[2]
            if succ.get(newp): continue     # new parent must still be a track end
            drop.add((p_, c)); add.append({"source_id": newp, "target_id": c, "edge_prob": None, "distance_um": 0.0})
            succ[newp].append(c)
    kept = [e for e in edges if (int(e["source_id"]), int(e["target_id"])) not in drop] + add
    return run_filter(nodes, kept, 7)

def steal_parent(r, ctx, rmax):
    """Return the pre-filter track end that the stolen child should attach to (recomputed from the graph)."""
    return None  # resolved inside rule via r fields (parent id of end not stored) -> use steal_end map

# Precompute, per stem, the nearest pre-filter track end for each fork child (id), to support 'steal' repairs.
STEAL_END = {}
try:
    for s in STEMS:
        nodes_pre, edges_pre = CACHE[s][0], CACHE[s][1]
        succ_pre = collections.defaultdict(list)
        for e in edges_pre: succ_pre[int(e["source_id"])].append(int(e["target_id"]))
        ends_by_t = collections.defaultdict(list)
        for n, v in nodes_pre.items():
            if not succ_pre.get(n): ends_by_t[v["t"]].append(n)
        pos = lambda n: np.array([nodes_pre[n]["z"], nodes_pre[n]["y"], nodes_pre[n]["x"]]) * VOX
        trees = {t: (cKDTree(np.array([pos(n) for n in ns])), ns) for t, ns in ends_by_t.items() if ns}
        m = {}
        for r in FT:
            if r["stem"] != s: continue
            for c in (r["c_steal"],):
                tr = trees.get(r["t"])
                if tr is None: continue
                dists, idxs = tr[0].query(pos(c), k=min(3, len(tr[1])))
                for dd, ii in zip(np.atleast_1d(dists), np.atleast_1d(idxs)):
                    if tr[1][ii] != r["parent"]:
                        m[(r["parent"], c)] = (tr[1][ii], float(dd)); break
        STEAL_END[s] = m
except Exception:
    traceback.print_exc()

def mk_steal(rmax):
    def rule(r, ctx):
        e = STEAL_END.get(r["stem"], {}).get((r["parent"], r["c_steal"]))
        if e and e[1] <= rmax:
            return ("steal", r["c_steal"], e[0])
        return None
    return rule

def _nan_child(r, c):
    ch = c["succ"][r["parent"]]
    if len(ch) < 2 or not r["nan_any"] or (r["p1"] < 0 and r["p2"] < 0):
        return None
    return ("drop", ch[0] if r["p1"] < 0 else ch[1])


DIV_RULES = {
    "drop ALL forks (keep longer branch)": lambda r, c: ("drop", r["c_short"]),
    "drop NaN child (safe-div/relink born)": _nan_child,
    "drop short branch down<=2 & pmin<0.8": lambda r, c: ("drop", r["c_short"]) if r["down_min"] <= 2 and (r["pmin"] < 0.8) else None,
    "drop short branch down<=3": lambda r, c: ("drop", r["c_short"]) if r["down_min"] <= 3 else None,
    "steal repair r<=4um": mk_steal(4.0),
    "steal repair r<=6um": mk_steal(6.0),
    "drop stolen child r<=4um": lambda r, c: ("drop", r["c_steal"]) if r["steal_min"] <= 4.0 else None,
    "drop fork if up<=2 (fresh parent)": lambda r, c: ("drop", r["c_short"]) if r["up"] <= 2 else None,
    "drop fork sister>=12um": lambda r, c: ("drop", r["c_short"]) if r["sister"] >= 12 else None,
}
try:
    off_res = []
    for label, rule in DIV_RULES.items():
        t1 = _t.time()
        G = {s: repaired_graph(s, rule) for s in STEMS}
        rows_off, _ = official_eval(G)
        OFF_ROWS[label] = rows_off
        s = off_summary(rows_off); b = off_summary(off_l7)
        # cell8 too (edges) for CI
        rows_c8 = [score_graph(s_, *G[s_]) for s_ in STEMS]
        rep = report("[div] " + label, rows_c8, base_rows)
        log(f"      OFFICIAL {label}: score={s['score']:.5f} ({s['score']-b['score']:+.5f}) adj={s['adj_edge_jaccard']:.5f} ({s['adj_edge_jaccard']-b['adj_edge_jaccard']:+.5f}) "
            f"div {s['division_tp']}/{s['division_fp']}/{s['division_fn']} divJ={s['division_jaccard']:.4f} ({s['division_jaccard']-b['division_jaccard']:+.4f}) | {_t.time()-t1:.0f}s")
        off_res.append({"label": label, "score": s["score"], "d_score": s["score"] - b["score"], "adj": s["adj_edge_jaccard"],
                        "div": [s["division_tp"], s["division_fp"], s["division_fn"]]})
    (OUT / "official_div_rules.json").write_text(_json.dumps(off_res, indent=1))
except Exception:
    traceback.print_exc()

# ================================================================ save
with (OUT / "results.json").open("w") as f:
    _json.dump(RESULTS, f, indent=1)
rows_all = []
for label, rows in ALL_ROWS.items():
    for r in rows:
        rows_all.append({"rule": label, **{k: v for k, v in r.items()}})
with (OUT / "rule_rows.csv").open("w", newline="") as f:
    keys = sorted({k for r in rows_all for k in r})
    w = _csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows_all)
with (OUT / "official_rows.json").open("w") as f:
    _json.dump(OFF_ROWS, f, indent=1)
log("CPU LAB done")
