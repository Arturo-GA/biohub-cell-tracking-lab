# ---- PRUNING LAB capture: record the graph that enters filter_short_track_components for every
# dataset, then run the normal single-pass writer (its output = x138 L6 baseline on these videos).
LAB_CAPTURED: dict[str, tuple[dict, list]] = {}
LAB_ORIG_FILTER = filter_short_track_components
_LAB_ORIG_FOG = filter_output_graph
_LAB_CUR = {"ds": None}


def filter_output_graph(*args, **kwargs):
    _LAB_CUR["ds"] = kwargs.get("dataset")
    return _LAB_ORIG_FOG(*args, **kwargs)


def filter_short_track_components(nodes_by_id, edges, stats):
    LAB_CAPTURED[_LAB_CUR["ds"]] = (
        {int(k): dict(v) for k, v in nodes_by_id.items()},
        [dict(e) for e in edges],
    )
    return LAB_ORIG_FILTER(nodes_by_id, edges, stats)


write_test_submission("base")
print("LAB captured pre-filter graphs:", len(LAB_CAPTURED), sorted(LAB_CAPTURED)[:6], flush=True)
