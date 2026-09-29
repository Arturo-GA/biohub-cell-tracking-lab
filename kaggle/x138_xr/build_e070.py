"""Build frozen E070 CPU labs and single-pass GPU submission candidates."""
import argparse
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import build
from e070_variants import VARIANTS, variant, LABS

W=Path(__file__).parent
ROOT=W.parents[1]


def lab(part):
    folder='k_e070_lab_'+part;slug='biohub-e070-lab-'+part
    nb=build.clean(build.SRC);cells=nb['cells']
    filt,official,_=build._lab_embed_common()
    text=build.src(cells[5])
    linefit=next(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='linefit_smooth_output_graph')
    base=(W/'cpu_lab7.py').read_text(encoding='utf-8').split('# Independently compare batched replay')[0]
    base=base.replace('OUT = WORKING_DIR / "cpu7"',f'OUT = WORKING_DIR / "e070_{part}"')
    for key,val in {'__OFFICIAL_FILES__':repr(official),'__FILTER_SRC__':repr(filt),'__LINEFIT_SRC__':repr(linefit),
        '__REPLAY_SRC__':repr((W/'replay_linefit.py').read_text()),'__XR_SRC__':repr((W/'extra_rules.py').read_text()),
        '__C3_VARIANTS_SRC__':repr((W/'c3_variants.py').read_text())}.items():base=base.replace(key,val)
    runner=(W/'e070_runner.py').read_text()
    for key,file in [('__CONTEXT_SOURCE__','context_rules.py'),('__POSITIONS_SOURCE__','e070_positions.py'),('__VARIANTS_SOURCE__','e070_variants.py')]:
        runner=runner.replace(key,repr((W/file).read_text()))
    payload=base+'\n'+runner
    manifest=dict(experiment='E070',part=part,candidates=LABS[part],payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),gpu=False)
    expected={k:{} for k in ('C3_public0955','C3_fork8')}
    with (ROOT/'results/E068/cpu7_rows.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            if r['config'] in expected:expected[r['config']][r['stem']]={k:float(r[k]) for k in ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes','adj_edge_jaccard')}
    assert all(len(v)==199 for v in expected.values())
    source=f'LAB_PART={part!r}\nRUN_MANIFEST={manifest!r}\nEXPECTED_BASELINE={expected!r}\n'+payload
    ast.parse(source)
    pre=copy.deepcopy(cells[8]);build.set_src(pre,'import json\nimport math\nfrom pathlib import Path\nimport numpy as np\nVOXEL_SCALE_UM=(1.625,.40625,.40625)\n')
    main=copy.deepcopy(cells[8]);build.set_src(main,source)
    nb['cells']=[cells[0],cells[2],cells[3],pre,main]
    build.write_kernel(folder,slug,slug.replace('-',' '),nb,kernel_sources=[f'jarturo/biohub-prune-lab{i}' for i in range(4,9)],gpu=False)
    (W/folder/'expected_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PREPARED LAB',part)


def inference(name):
    cfg=variant(name);xr={"BIOHUB_XR_"+k.upper():str(v) for k,v in cfg.pop('xr').items()}
    overrides={'BIOHUB_OUTPUT_MIN_TRACK_LEN':'7','BIOHUB_REPAIR_DEADLINE_S':'34200'}
    if 'dc_thr' in cfg:overrides['BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD']=str(cfg['dc_thr'])
    slug='biohub-e070-'+name.lower().replace('_','-');folder='k_e070_'+name.lower()
    build.make_xr(folder,slug,slug.replace('-',' '),overrides,xr)
    path=W/folder/'notebook.ipynb';nb=json.loads(path.read_text())
    text=build.src(nb['cells'][5]);extra=(W/'context_rules.py').read_text()+'\n'+(W/'e070_positions.py').read_text()+'\n'
    if 'fork_smooth' in cfg or 'position_mode' in cfg:
        extra+='''
_e070_upstream_linefit = linefit_smooth_output_graph
def linefit_smooth_output_graph(nodes_by_id, edges, stats):
    raw = {k:dict(v) for k,v in nodes_by_id.items()}
    smoothed = _e070_upstream_linefit(nodes_by_id, edges, stats)
    try:
        output, changed = e070_positions(raw, edges, smoothed, E070_CFG)
        stats['e070_position_changes'] = changed
        return output
    except Exception as exc:
        print('REPAIR FAILED E070 positions:', type(exc).__name__, str(exc), flush=True)
        return smoothed
'''.replace('E070_CFG',repr(cfg))
    text=build.must_replace(text,'def filter_output_graph(',extra+'\ndef filter_output_graph(')
    pipeline=build.XR_NEW_CALL
    if cfg.get('swap'):
        pipeline+='''    try:
        nodes_by_id, edges, _e070_stats = temporal_swaps(nodes_by_id, edges, relative_gain=E070_GAIN)
        stats.update(_e070_stats)
    except Exception as exc:
        print('REPAIR FAILED E070 swaps:', type(exc).__name__, str(exc), flush=True)
'''.replace('E070_GAIN',repr(cfg['swap']))
    pipeline+='    print(f"  [{dataset}] E070 GRAPH: " + str({k:v for k,v in stats.items() if k in ("swap_pairs", "xr_prune_nodes_removed", "xr_fork_drops")}), flush=True)\n'
    text=build.must_replace(text,build.XR_NEW_CALL,pipeline)
    build.set_src(nb['cells'][5],text)
    code='\n'.join(build.src(c) for c in nb['cells'] if c['cell_type']=='code')
    manifest=dict(candidate=name,experiment='E070',base='c3_public_0.955',fixed=overrides,xr=xr,config=cfg,source_sha256=hashlib.sha256(code.encode()).hexdigest())
    marker='import json as _manifest_json\nfrom pathlib import Path as _ManifestPath\n_manifest = '+repr(manifest)+'\n_ManifestPath("/kaggle/working/run_manifest.json").write_text(_manifest_json.dumps(_manifest,sort_keys=True))\nprint("RUN_MANIFEST",_manifest_json.dumps(_manifest,sort_keys=True))\n'
    nb['cells'].append(dict(cell_type='code',metadata={},outputs=[],execution_count=None,source=marker.splitlines(keepends=True)))
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(build.src(c))
    path.write_text(json.dumps(nb,indent=1))
    (path.parent/'expected_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    mp=path.parent/'kernel-metadata.json';meta=json.loads(mp.read_text());meta['machine_shape']='NvidiaTeslaT4';mp.write_text(json.dumps(meta,indent=2)+'\n')
    print('PREPARED INFERENCE',name)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['lab','inference']);p.add_argument('name');a=p.parse_args()
    if a.kind=='lab':assert a.name in LABS;lab(a.name)
    else:assert a.name in VARIANTS;inference(a.name)
