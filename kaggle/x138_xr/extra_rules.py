
# ---- XR: extra graph rules shared by the submission kernel and the CPU lab. All default OFF (x138 output unchanged).
#   pre-filter  : cut long relink/gap edges (edge_prob None, distance > XR cut_nan_dist_um)
#                 cut low-confidence track-end edges (ILP prob < cut_end_prob; first/last/both)
#                 drop the shorter branch of forks whose shorter branch has < fork_min_branch nodes
#   post-filter : prune non-division components with size < S and mean ILP prob < P, per dataset prefix
#                 spec "44b6:8:0.7,6bba:11:0.7,*:8:0.7" ('*' = fallback)
import collections as _xr_col
import os as _xr_os

XR = {
    "cut_nan_dist_um": float(_xr_os.environ.get("BIOHUB_XR_CUT_NAN_DIST_UM", "0")),
    "cut_end_prob": _xr_os.environ.get("BIOHUB_XR_CUT_END_PROB", "0"),          # number or per-prefix spec
    "cut_end_mode": _xr_os.environ.get("BIOHUB_XR_CUT_END_MODE", "both").strip().lower(),
    "fork_min_branch": _xr_os.environ.get("BIOHUB_XR_FORK_MIN_BRANCH", "0"),    # number or per-prefix spec
    "prune": _xr_os.environ.get("BIOHUB_XR_PRUNE", "").strip(),
    "cut_end_iter": int(_xr_os.environ.get("BIOHUB_XR_CUT_END_ITER", "1")),   # peel low-confidence end edges this many times
    "fork_nan_only": _xr_os.environ.get("BIOHUB_XR_FORK_NAN_ONLY", "0"),      # 1: only drop safe-div-born (edge_prob None) short branches
    "fork_preserve_boundary": _xr_os.environ.get("BIOHUB_XR_FORK_PRESERVE_BOUNDARY", "0"),
}


def xr_configure(**kw):
    XR.update(kw)


def _xr_prob(e):
    p = e.get("edge_prob")
    if p is None:
        return None
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    return p if p == p else None


def _xr_adj(edges):
    succ = _xr_col.defaultdict(list)
    pred = _xr_col.defaultdict(list)
    for e in edges:
        succ[int(e["source_id"])].append(e)
        pred[int(e["target_id"])].append(e)
    return succ, pred


def _xr_prune_params(dataset):
    spec = XR["prune"]
    if not spec:
        return None
    pfx = (dataset or "").split("_")[0]
    fallback = None
    for item in spec.split(","):
        parts = item.strip().split(":")
        if len(parts) != 3:
            continue
        key, S, P = parts
        if key == pfx:
            return int(S), float(P)
        if key == "*":
            fallback = (int(S), float(P))
    return fallback


def _xr_scalar(value, dataset, cast=float, default=0):
    """A plain number applies to every dataset; a spec like '44b6:0.4,6bba:0.5,*:0.5' is resolved by prefix (0 = off)."""
    if isinstance(value, (int, float)):
        return cast(value)
    text = str(value).strip()
    if not text:
        return cast(0)
    if ":" not in text:
        return cast(float(text))
    pfx = (dataset or "").split("_")[0]
    fallback = cast(default)
    for item in text.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2:
            continue
        key, v = parts
        if key == pfx:
            return cast(float(v))
        if key == "*":
            fallback = cast(float(v))
    return fallback


# Snapshot the configured defaults once. An unknown prefix must not inherit the
# preceding video's override or turn off a safety threshold by returning zero.
_XR_BASE_TAU = float(globals().get("SAFE_DIV_SISTER_SYMMETRY_TAU", _xr_os.environ.get("BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU", "0.6")))
_XR_BASE_DC = float(globals().get("DEEPCENTER_SAFE_DIV_THRESHOLD", _xr_os.environ.get("BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD", "0.25")))


def xr_set_safe_div_params(dataset=None):
    """Per-embryo safe-division thresholds (read before add_safe_divisions_postlink runs).
    BIOHUB_XR_TAU_SPEC e.g. '44b6:0.6,6bba:1.0' sets SAFE_DIV_SISTER_SYMMETRY_TAU; BIOHUB_XR_DC_SPEC sets
    DEEPCENTER_SAFE_DIV_THRESHOLD the same way. Empty spec = leave x138's value untouched."""
    try:
        g = globals()
        tau_spec = _xr_os.environ.get("BIOHUB_XR_TAU_SPEC", "").strip()
        if tau_spec:
            g["SAFE_DIV_SISTER_SYMMETRY_TAU"] = _xr_scalar(tau_spec, dataset, default=_XR_BASE_TAU)
        dc_spec = _xr_os.environ.get("BIOHUB_XR_DC_SPEC", "").strip()
        if dc_spec:
            g["DEEPCENTER_SAFE_DIV_THRESHOLD"] = _xr_scalar(dc_spec, dataset, default=_XR_BASE_DC)
        if tau_spec or dc_spec:
            print(f"  [{dataset}] safe-div params: tau={g.get('SAFE_DIV_SISTER_SYMMETRY_TAU')} dc={g.get('DEEPCENTER_SAFE_DIV_THRESHOLD')}", flush=True)
    except Exception as exc:
        print(f"  xr_set_safe_div_params skipped (non-fatal): {type(exc).__name__}: {exc}")


def xr_pre_filter(nodes_by_id, edges, dataset=None, stats=None):
    if stats is None:
        stats = {}
    try:
        if XR["cut_nan_dist_um"] > 0:
            before = len(edges)
            thr = XR["cut_nan_dist_um"]
            edges = [e for e in edges if not (_xr_prob(e) is None and float(e.get("distance_um", 0.0) or 0.0) > thr)]
            stats["xr_cut_nan_edges"] = before - len(edges)
        end_thr = _xr_scalar(XR["cut_end_prob"], dataset)
        if end_thr > 0:
            mode, thr = XR["cut_end_mode"], end_thr
            total_cut = 0
            for _round in range(max(1, int(XR.get("cut_end_iter", 1)))):
                succ, pred = _xr_adj(edges)
                keep, cut = [], 0
                for e in edges:
                    p = _xr_prob(e)
                    if p is not None and p < thr:
                        s, d = int(e["source_id"]), int(e["target_id"])
                        is_first = not pred.get(s)
                        is_last = not succ.get(d)
                        if (mode in ("both", "first") and is_first) or (mode in ("both", "last") and is_last):
                            cut += 1
                            continue
                    keep.append(e)
                edges = keep
                total_cut += cut
                if cut == 0:
                    break
            stats["xr_cut_end_edges"] = total_cut
        minb = _xr_scalar(XR["fork_min_branch"], dataset, cast=int)
        if minb > 0:
            succ, pred = _xr_adj(edges)

            def down_len(n, cap=64):
                k = 1
                while k < cap:
                    nxt = succ.get(n, [])
                    if len(nxt) != 1:
                        return k
                    m = int(nxt[0]["target_id"])
                    if len(pred.get(m, [])) != 1:
                        return k
                    n = m
                    k += 1
                return k

            nan_only = str(XR.get("fork_nan_only", "0")).strip() not in ("0", "", "false", "False")
            preserve_boundary = str(XR.get("fork_preserve_boundary", "0")).strip() not in ("0", "", "false", "False")
            last_frame = max((int(v["t"]) for v in nodes_by_id.values()), default=-1) if preserve_boundary else 0
            drop = set()
            for s, outs in succ.items():
                if len(outs) < 2:
                    continue
                lens = sorted((down_len(int(e["target_id"])), i) for i, e in enumerate(outs))
                if lens[0][0] < minb:
                    e = outs[lens[0][1]]
                    if nan_only and _xr_prob(e) is not None:
                        continue   # only revert branches added by the post-process (edge_prob None), never ILP edges
                    if preserve_boundary:
                        child_t = int(nodes_by_id[int(e["target_id"])]["t"])
                        if last_frame - child_t + 1 < minb:
                            continue  # insufficient remaining observation to judge this branch short
                    drop.add((int(e["source_id"]), int(e["target_id"])))
            if drop:
                edges = [e for e in edges if (int(e["source_id"]), int(e["target_id"])) not in drop]
            stats["xr_fork_drops"] = len(drop)
    except Exception as exc:  # optional stage: never change the fallback behaviour
        print(f"  xr_pre_filter skipped (non-fatal): {type(exc).__name__}: {exc}")
    return nodes_by_id, edges


def xr_post_filter(nodes_by_id, edges, dataset=None, stats=None):
    if stats is None:
        stats = {}
    params = _xr_prune_params(dataset)
    if params is None or not edges or not nodes_by_id:
        return nodes_by_id, edges
    try:
        S, P = params
        parent = {n: n for n in nodes_by_id}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        out_count = _xr_col.Counter()
        for e in edges:
            s, d = int(e["source_id"]), int(e["target_id"])
            out_count[s] += 1
            if s in parent and d in parent:
                rs, rd = find(s), find(d)
                if rs != rd:
                    parent[rs] = rd
        comps = _xr_col.defaultdict(list)
        for n in nodes_by_id:
            comps[find(n)].append(n)
        probs = _xr_col.defaultdict(list)
        for e in edges:
            p = _xr_prob(e)
            if p is not None:
                probs[find(int(e["source_id"]))].append(p)
        remove = set()
        for r, members in comps.items():
            if len(members) >= S or any(out_count.get(m, 0) >= 2 for m in members):
                continue
            mp = (sum(probs[r]) / len(probs[r])) if probs[r] else 0.0
            if mp < P:
                remove.update(members)
        if not remove:
            return nodes_by_id, edges
        kept = {k: v for k, v in nodes_by_id.items() if k not in remove}
        kept_edges = [e for e in edges if int(e["source_id"]) in kept and int(e["target_id"]) in kept]
        stats["xr_prune_nodes_removed"] = len(nodes_by_id) - len(kept)
        stats["xr_prune_edges_removed"] = len(edges) - len(kept_edges)
        return kept, kept_edges
    except Exception as exc:
        print(f"  xr_post_filter skipped (non-fatal): {type(exc).__name__}: {exc}")
        return nodes_by_id, edges
