"""Build a fixed, single-pass c3 submission with a CPU6-tested hypothesis."""
import argparse
import ast
import hashlib
import json

import build


VARIANTS = {
    'temporal_swaps_strict': {'swap': .35},
    'temporal_swaps_balanced': {'swap': .20},
    'context_end': {'cut_context': 'end'},
    'context_long': {'cut_context': 'long'},
    'context_both': {'cut_context': 'both'},
    'fork_smooth03': {'smooth_weight': .3},
    'fork_smooth0': {'smooth_weight': 0.},
    'swaps_context': {'swap': .35, 'cut_context': 'both'},
    'context_smooth': {'cut_context': 'both', 'smooth_weight': .3},
}


def make(variant):
    cfg=VARIANTS[variant]
    slug='biohub-c3-'+variant.replace('_','-')
    folder='k_c3_'+variant
    overrides={'BIOHUB_OUTPUT_MIN_TRACK_LEN':'7','BIOHUB_REPAIR_DEADLINE_S':'34200'}
    xr={'BIOHUB_XR_PRUNE':'44b6:8:0.7,6bba:11:0.7','BIOHUB_XR_CUT_NAN_DIST_UM':'8',
        'BIOHUB_XR_CUT_END_PROB':'0.5','BIOHUB_XR_CUT_END_MODE':'last'}
    build.make_xr(folder,slug,'Biohub c3 '+variant.replace('_',' '),overrides,xr)
    path=build.W/folder/'notebook.ipynb'
    nb=json.loads(path.read_text(encoding='utf-8'))
    c5=build.src(nb['cells'][5])
    extra=(build.W/'context_rules.py').read_text(encoding='utf-8')
    if 'smooth_weight' in cfg:
        extra+='''
_context_original_linefit = linefit_smooth_output_graph
def linefit_smooth_output_graph(nodes_by_id, edges, stats):
    raw = {k:dict(v) for k,v in nodes_by_id.items()}
    smoothed = _context_original_linefit(nodes_by_id, edges, stats)
    out, changed = topology_smoothing(raw, edges, smoothed, weight=WEIGHT)
    stats["context_smoothing_nodes"] = changed
    return out
'''.replace('WEIGHT',repr(cfg['smooth_weight']))
    c5=build.must_replace(c5,'def filter_output_graph(',extra+'\ndef filter_output_graph(')
    if cfg.get('cut_context'):
        pre=f'    nodes_by_id, edges, _ctx_stats = context_cuts(nodes_by_id, edges, protect={cfg["cut_context"]!r})\n    stats.update(_ctx_stats)\n'
    else:
        pre='    nodes_by_id, edges = xr_pre_filter(nodes_by_id, edges, dataset, stats)\n'
    pipeline=pre+'    nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats)\n    nodes_by_id, edges = xr_post_filter(nodes_by_id, edges, dataset, stats)\n'
    if cfg.get('swap'):
        pipeline+=f'    nodes_by_id, edges, _ctx_stats = temporal_swaps(nodes_by_id, edges, relative_gain={cfg["swap"]!r})\n    stats.update(_ctx_stats)\n'
    pipeline+='    print(f"  [{dataset}] CONTEXT: " + str({k:v for k,v in stats.items() if k in ("swap_pairs", "swap_candidates", "long_saved", "end_saved")}), flush=True)\n'
    c5=build.must_replace(c5,build.XR_NEW_CALL,pipeline)
    build.set_src(nb['cells'][5],c5)
    code='\n'.join(build.src(c) for c in nb['cells'] if c['cell_type']=='code')
    manifest=dict(candidate=variant,base='c3_public_0.955',context=cfg,
                  source_sha256=hashlib.sha256(code.encode()).hexdigest(),fixed=overrides,xr=xr)
    marker='import json as _manifest_json\nfrom pathlib import Path as _ManifestPath\n_manifest = '+repr(manifest)+'\n_ManifestPath("/kaggle/working/run_manifest.json").write_text(_manifest_json.dumps(_manifest, sort_keys=True))\nprint("RUN_MANIFEST", _manifest_json.dumps(_manifest, sort_keys=True))\n'
    nb['cells'].append(dict(cell_type='code',metadata={},outputs=[],execution_count=None,source=marker.splitlines(keepends=True)))
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(build.src(c))
    path.write_text(json.dumps(nb,indent=1),encoding='utf-8')
    (path.parent/'expected_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('PREPARED',variant,manifest['source_sha256'])
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('variant',choices=list(VARIANTS)+['all'])
    args=p.parse_args()
    for variant in VARIANTS if args.variant=='all' else [args.variant]:make(variant)
