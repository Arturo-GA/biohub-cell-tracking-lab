# ---- PRUNING LAB evaluation: score many short-component pruning rules on held-out TRAIN videos
# with the validator's metric reimplementation (cell 8). Pruning never touches division components.
import collections
import csv as _csv
import json as _json
import time as _t

import numpy as np

VALIDATOR_MATCH_RADIUS_UM = 7.0
VALIDATOR_NODE_COUNT_PENALTY_A = 0.1
VALIDATOR_DIVISION_WEIGHT = 0.1
LAB_VAL8_CHECK = ['44b6_12dfb391', '44b6_267148e4', '44b6_2a2eff9f', '44b6_341df25f',
                  '6bba_062c8d37', '6bba_07e24132', '6bba_085bf656', '6bba_09961292']
LAB_TRAIN_DIR = COMP_DIR / "train"
LAB_OUT = WORKING_DIR / "lab"
(LAB_OUT / "graphs").mkdir(parents=True, exist_ok=True)
_t0 = _t.time()

LAB_GT = {}
for stem in sorted(LAB_CAPTURED):
    gpath = LAB_TRAIN_DIR / f"{stem}.geff"
    gn, ge = graph_to_plain(graph_from_geff(gpath))
    LAB_GT[stem] = (gn, ge, read_estimated_true_node_count(gpath))
print(f"LAB GT loaded for {len(LAB_GT)} videos | {_t.time() - _t0:.0f}s", flush=True)

# Score on what the LB sees: x138 runs linefit_smooth_output_graph after the short-track filter and the
# writer rounds z/y/x. Pruning removes whole components, so smoothing the full captured graph once is exact.
LAB_PLAIN, LAB_PLAINF = {}, {}
for _stem in sorted(LAB_CAPTURED):
    _n, _e = LAB_CAPTURED[_stem]
    _sm = linefit_smooth_output_graph({k: dict(v) for k, v in _n.items()}, _e, collections.Counter())
    LAB_PLAINF[_stem] = {k: (int(v["t"]), float(v["z"]), float(v["y"]), float(v["x"])) for k, v in _sm.items()}
    LAB_PLAIN[_stem] = {k: (int(v["t"]), float(max(0, int(round(float(v["z"]))))),
                            float(max(0, int(round(float(v["y"]))))), float(max(0, int(round(float(v["x"])))))) for k, v in _sm.items()}
print(f"LAB linefit+round done | {_t.time() - _t0:.0f}s", flush=True)


def lab_components(nodes, edges):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    out_count = collections.Counter()
    for e in edges:
        s, d = int(e["source_id"]), int(e["target_id"])
        out_count[s] += 1
        if s in parent and d in parent:
            rs, rd = find(s), find(d)
            if rs != rd:
                parent[rs] = rd
    comps = collections.defaultdict(list)
    for n in nodes:
        comps[find(n)].append(n)
    return comps, out_count


def lab_apply_L(nodes, edges, L):
    g = globals()
    saved = g["OUTPUT_MIN_TRACK_LEN"]
    g["OUTPUT_MIN_TRACK_LEN"] = L
    try:
        return LAB_ORIG_FILTER(nodes, edges, collections.Counter())
    finally:
        g["OUTPUT_MIN_TRACK_LEN"] = saved


def lab_mean_prob(members_set, edges):
    probs = []
    for e in edges:
        if int(e["source_id"]) in members_set:
            p = e.get("edge_prob")
            try:
                p = float(p)
            except (TypeError, ValueError):
                continue
            if np.isfinite(p):
                probs.append(p)
    return float(np.mean(probs)) if probs else 0.0


def lab_extra_prune(nodes, edges, maxsize, mode, t_lo, t_hi, prob_thr=None, minsize=0):
    """Remove non-division components with < maxsize nodes, subject to a border/prob exemption."""
    comps, out_count = lab_components(nodes, edges)
    remove = set()
    for members in comps.values():
        if len(members) >= maxsize or len(members) < minsize:
            continue
        if any(out_count.get(m, 0) >= 2 for m in members):
            continue
        ts = [int(nodes[m]["t"]) for m in members]
        starts, ends = min(ts) <= t_lo, max(ts) >= t_hi
        if mode == "interior" and (starts or ends):
            continue
        if mode == "keep_start" and starts:
            continue
        if mode == "keep_end" and ends:
            continue
        if mode == "prob" and lab_mean_prob(set(members), edges) >= prob_thr:
            continue
        remove.update(members)
    if not remove:
        return nodes, edges
    kn = {k: v for k, v in nodes.items() if k not in remove}
    ke = [e for e in edges if int(e["source_id"]) in kn and int(e["target_id"]) in kn]
    return kn, ke


RULES = [("L", L, None, None) for L in (6, 7, 8, 9, 10, 11, 12)]
RULES += [("L7+" + m, 7, s, m) for m in ("interior", "keep_start", "keep_end") for s in (8, 9, 10, 12)]
RULES += [("L6+interior", 6, s, "interior") for s in (7, 8, 9)]
RULES += [("L7+prob", 7, s, ("prob", p)) for s in (9, 12) for p in (0.5, 0.8)]
RULES += [("L7+interior_min7", 7, s, "interior_min7") for s in (8, 9, 10)]


def lab_run_rule(stem, rule):
    name, base_L, maxsize, mode = rule
    nodes, edges = LAB_CAPTURED[stem]
    n2, e2 = lab_apply_L(nodes, edges, base_L)
    if maxsize is not None:
        ts = [int(v["t"]) for v in nodes.values()]
        t_lo, t_hi = min(ts), max(ts)
        if isinstance(mode, tuple):
            n2, e2 = lab_extra_prune(n2, e2, maxsize, "prob", t_lo, t_hi, prob_thr=mode[1])
        elif mode == "interior_min7":
            n2, e2 = lab_extra_prune(n2, e2, maxsize, "interior", t_lo, t_hi, minsize=7)
        else:
            n2, e2 = lab_extra_prune(n2, e2, maxsize, mode, t_lo, t_hi)
    return n2, e2


def lab_rule_label(rule):
    name, base_L, maxsize, mode = rule
    if maxsize is None:
        return f"L{base_L}"
    m = f"prob{mode[1]}" if isinstance(mode, tuple) else mode
    return f"L{base_L}+{m}<{maxsize}"


lab_rows = []
for rule in RULES:
    label = lab_rule_label(rule)
    rows = []
    for stem in sorted(LAB_CAPTURED):
        n2, e2 = lab_run_rule(stem, rule)
        gn, ge, tt = LAB_GT[stem]
        row = score_sample({k: LAB_PLAIN[stem][k] for k in n2}, [(int(e["source_id"]), int(e["target_id"])) for e in e2], gn, ge, tt)
        row["stem"] = stem
        row["rule"] = label
        rows.append(row)
    lab_rows.extend(rows)
    summ = aggregate_official(rows)
    if label == "L6":
        _v8 = [r for r in rows if r["stem"] in set(LAB_VAL8_CHECK)]
        if _v8:
            _fl = []
            for r in _v8:
                n2, e2 = lab_run_rule(r["stem"], rule)
                gn, ge, tt = LAB_GT[r["stem"]]
                _fl.append(score_sample({k: LAB_PLAINF[r["stem"]][k] for k in n2}, [(int(e["source_id"]), int(e["target_id"])) for e in e2], gn, ge, tt))
            print(f"SANITY val8 L6: rounded proxy={aggregate_official(_v8)['proxy_score']:.5f} "
                  f"float proxy={aggregate_official(_fl)['proxy_score']:.5f} (x138 in-kernel validator base was 0.9555)", flush=True)
    by_pfx = {}
    for pfx in sorted({r["stem"].split("_")[0] for r in rows}):
        by_pfx[pfx] = aggregate_official([r for r in rows if r["stem"].startswith(pfx)])["proxy_score"]
    tp = sum(r["edge_tp"] for r in rows); fp = sum(r["edge_fp"] for r in rows); tpred = sum(r["t_pred"] for r in rows)
    print(f"[{label:<22}] proxy={summ['proxy_score']:.5f} adj={summ['adjusted_edge_jaccard']:.5f} "
          f"divJ={summ['division_jaccard']:.4f} TP={tp} FP={fp} Tpred={tpred} "
          + " ".join(f"{k}={v:.5f}" for k, v in by_pfx.items()) + f" | {_t.time() - _t0:.0f}s", flush=True)

with (LAB_OUT / "lab_rows.csv").open("w", newline="") as f:
    keys = sorted({k for r in lab_rows for k in r})
    w = _csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(lab_rows)

# per-component feature table on the pre-filter graph (components up to 20 nodes)
comp_rows = []
for stem in sorted(LAB_CAPTURED):
    nodes, edges = LAB_CAPTURED[stem]
    gn, ge, tt = LAB_GT[stem]
    plain = LAB_PLAIN[stem]
    p2g, g2p = match_nodes_bipartite(plain, gn, max_dist=VALIDATOR_MATCH_RADIUS_UM)
    gt_out = collections.defaultdict(set); gt_in = {}
    for s, d in ge:
        gt_out[s].add(d); gt_in[d] = s
    comps, out_count = lab_components(nodes, edges)
    ts_all = [int(v["t"]) for v in nodes.values()]
    t_lo, t_hi = min(ts_all), max(ts_all)
    zs_all = [float(v["z"]) for v in nodes.values()]
    z_lo, z_hi = min(zs_all), max(zs_all)
    root_of = {m: r for r, ms in comps.items() for m in ms}
    ecount = collections.defaultdict(lambda: [0, 0, 0, []])
    for e in edges:
        s, d = int(e["source_id"]), int(e["target_id"])
        ms, md = p2g.get(s), p2g.get(d)
        tp = ms is not None and md is not None and md in gt_out.get(ms, ())
        fp = (not tp) and ((md is not None and md in gt_in) or (ms is not None and bool(gt_out.get(ms))))
        c = ecount[root_of.get(s)]
        c[0] += int(tp); c[1] += int(fp); c[2] += 1
        try:
            p = float(e.get("edge_prob"))
            if np.isfinite(p):
                c[3].append(p)
        except (TypeError, ValueError):
            pass
    for r, members in comps.items():
        if len(members) > 20:
            continue
        ts = [int(nodes[m]["t"]) for m in members]
        zs = [float(nodes[m]["z"]) for m in members]
        c = ecount.get(r, [0, 0, 0, []])
        comp_rows.append({
            "stem": stem, "size": len(members), "t0": min(ts), "t1": max(ts), "t_lo": t_lo, "t_hi": t_hi,
            "touch_start": int(min(ts) <= t_lo), "touch_end": int(max(ts) >= t_hi),
            "z_mean": float(np.mean(zs)), "z_lo": z_lo, "z_hi": z_hi,
            "division": int(any(out_count.get(m, 0) >= 2 for m in members)),
            "n_edges": c[2], "tp_edges": c[0], "fp_edges": c[1],
            "mean_prob": float(np.mean(c[3])) if c[3] else -1.0, "n_prob": len(c[3]),
            "matched_nodes": sum(1 for m in members if m in p2g),
        })
with (LAB_OUT / "lab_components.csv").open("w", newline="") as f:
    w = _csv.DictWriter(f, fieldnames=list(comp_rows[0]))
    w.writeheader()
    w.writerows(comp_rows)

# raw graphs + GT for offline re-evaluation without GPU
for stem in sorted(LAB_CAPTURED):
    nodes, edges = LAB_CAPTURED[stem]
    gn, ge, tt = LAB_GT[stem]
    nid = np.array(sorted(nodes), dtype=np.int64)
    narr = np.array([[nodes[i]["t"], nodes[i]["z"], nodes[i]["y"], nodes[i]["x"]] for i in nid], dtype=np.float32)
    def _p(e):
        try:
            v = float(e.get("edge_prob"))
            return v if np.isfinite(v) else np.nan
        except (TypeError, ValueError):
            return np.nan
    earr = np.array([[int(e["source_id"]), int(e["target_id"])] for e in edges], dtype=np.int64).reshape(-1, 2)
    eprob = np.array([_p(e) for e in edges], dtype=np.float32)
    gid = np.array(sorted(gn), dtype=np.int64)
    garr = np.array([gn[i] for i in gid], dtype=np.float32).reshape(-1, 4)
    gearr = np.array(ge, dtype=np.int64).reshape(-1, 2)
    nfin = np.array([LAB_PLAIN[stem][int(i)] for i in nid], dtype=np.float32)
    # provenance flags (bitmasks) so post-process stages can be attributed offline
    _EF = (("motion_relinked", 1), ("gap_closed", 2), ("gap2_recovered", 4), ("gap_synthetic", 8), ("gap_filled", 16), ("safe_division", 32))
    _NF = (("readmitted", 1), ("gap_synthetic", 2), ("gapfill_peak", 4))
    eflags = np.array([sum(b for k, b in _EF if e.get(k)) for e in edges], dtype=np.int16)
    nflags = np.array([sum(b for k, b in _NF if nodes[int(i)].get(k)) for i in nid], dtype=np.int16)
    _sdt = globals().get("SAFEDIV_TRACE", {}).get(stem)
    _sdt = np.zeros((0, 14), dtype=np.float64) if _sdt is None else _sdt
    np.savez_compressed(LAB_OUT / "graphs" / f"{stem}.npz", nid=nid, nodes=narr, nodes_final=nfin, edges=earr, eprob=eprob,
                        eflags=eflags, nflags=nflags, sdtrace=_sdt,
                        gid=gid, gnodes=garr, gedges=gearr, t_true=np.array([tt if tt is not None else -1.0]))

print("LAB done:", len(lab_rows), "rows,", len(comp_rows), "components", f"| {_t.time() - _t0:.0f}s", flush=True)
import shutil as _sh
_sh.rmtree(REPO_DIR / "predictions", ignore_errors=True)
_sh.rmtree(WORKING_DIR / "edge_cache", ignore_errors=True)
