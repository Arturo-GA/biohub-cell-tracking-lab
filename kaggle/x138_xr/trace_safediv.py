# ---- SAFE-DIV TRACE: record every wide-gate division candidate (with the features x138 uses to accept or
# reject it and the DeepCenter score) so the decision thresholds can be replayed offline. The original
# add_safe_divisions_postlink still runs unchanged afterwards.
SAFEDIV_TRACE: dict[str, np.ndarray] = {}
SDT_COLS = ["t", "source_id", "child_id", "cand_id", "child_dist", "parent_dist", "sister_dist", "mutual_nn",
            "div_ok", "diverge", "symmetry", "dc_cand", "dc_child", "added"]
_SDT_ORIG = add_safe_divisions_postlink


def _sdt_enumerate(nodes_by_id, edges, dataset, deepcenter_bundle, frame_cache, deepcenter_cache):
    out_by_source: dict[int, list] = {}
    incoming: set[int] = set()
    for edge in edges:
        out_by_source.setdefault(int(edge["source_id"]), []).append(edge)
        incoming.add(int(edge["target_id"]))
    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)
    rows = []
    dc_memo: dict[int, float] = {}

    def dc(node_id):
        if node_id in dc_memo:
            return dc_memo[node_id]
        node = nodes_by_id[node_id]
        try:
            s = deepcenter_score_point(dataset, int(node["t"]), node_point(node), deepcenter_bundle, frame_cache, deepcenter_cache)
        except Exception:
            s = None
        v = float("nan") if s is None else float(s)
        dc_memo[node_id] = v
        return v

    for t in sorted(ids_by_t):
        child_ids = ids_by_t.get(t + 1, [])
        if not child_ids:
            continue
        source_ids = [n for n in ids_by_t[t] if len(out_by_source.get(n, [])) == 1]
        cand_ids = [n for n in child_ids if n not in incoming]
        if not source_ids or not cand_ids:
            continue
        cand_pos = np.stack([_position_um(nodes_by_id[c]) for c in cand_ids])
        tree = cKDTree(cand_pos)
        for source_id in source_ids:
            source = nodes_by_id[source_id]
            ex_edge = out_by_source[source_id][0]
            ex_id = int(ex_edge["target_id"])
            ex = nodes_by_id.get(ex_id)
            if ex is None or int(ex["t"]) != t + 1:
                continue
            child_dist = edge_distance_um(source, ex)
            if child_dist > 14.0:
                continue
            _, nn_idx = tree.query(_position_um(ex))
            mutual_nn_id = cand_ids[int(nn_idx)]
            near = tree.query_ball_point(_position_um(source), r=14.0)
            for j in near:
                cand_id = cand_ids[j]
                cand = nodes_by_id[cand_id]
                parent_dist = edge_distance_um(source, cand)
                sister_dist = edge_distance_um(ex, cand)
                if sister_dist > 20.0:
                    continue
                c1_succ = out_by_source.get(ex_id, [])
                q_succ = out_by_source.get(cand_id, [])
                div_ok, diverge = 0, float("nan")
                if len(c1_succ) == 1 and len(q_succ) == 1:
                    g1 = nodes_by_id.get(int(c1_succ[0]["target_id"]))
                    g2 = nodes_by_id.get(int(q_succ[0]["target_id"]))
                    if g1 is not None and g2 is not None and int(g1["t"]) == t + 2 and int(g2["t"]) == t + 2:
                        div_ok = 1
                        diverge = edge_distance_um(g1, g2) - sister_dist
                denom = max((child_dist + parent_dist) / 2.0, 1e-6)
                symmetry = abs(child_dist - parent_dist) / denom
                rows.append([t, source_id, ex_id, cand_id, child_dist, parent_dist, sister_dist, int(cand_id == mutual_nn_id),
                             div_ok, diverge, symmetry, dc(cand_id), dc(ex_id), 0.0])
    return rows


def add_safe_divisions_postlink(nodes_by_id, edges, stats, dataset=None, deepcenter_bundle=None, frame_cache=None, deepcenter_cache=None):
    rows = []
    frame_cache = frame_cache if frame_cache is not None else {}
    deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}
    try:
        if OUTPUT_SAFE_DIVISIONS and edges and nodes_by_id:
            rows = _sdt_enumerate(nodes_by_id, edges, dataset, deepcenter_bundle, frame_cache, deepcenter_cache)
    except Exception as exc:
        print(f"  [{dataset}] safe-div trace skipped (non-fatal): {type(exc).__name__}: {exc}", flush=True)
        rows = []
    n_before = len(edges)
    result = _SDT_ORIG(nodes_by_id, edges, stats, dataset=dataset, deepcenter_bundle=deepcenter_bundle,
                       frame_cache=frame_cache, deepcenter_cache=deepcenter_cache)
    try:
        added = {(int(e["source_id"]), int(e["target_id"])) for e in result[n_before:]}
        for r in rows:
            if (int(r[1]), int(r[3])) in added:
                r[13] = 1.0
        arr = np.array(rows, dtype=np.float64).reshape(-1, len(SDT_COLS))
        SAFEDIV_TRACE[str(dataset)] = arr
        print(f"  [{dataset}] safe-div trace: {len(rows)} candidates, {int(arr[:, 13].sum()) if len(rows) else 0} added by x138", flush=True)
    except Exception as exc:
        print(f"  [{dataset}] safe-div trace bookkeeping skipped: {type(exc).__name__}: {exc}", flush=True)
    return result
