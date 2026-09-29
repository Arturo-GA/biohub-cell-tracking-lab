"""Build E069 private CPU duplicate and GPU linker-cache experiments."""
import argparse
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import build

W=Path(__file__).parent
SOURCES=[f'jarturo/biohub-prune-lab{i}' for i in range(4,9)]


def make(mode):
    folder={'capture':'k_e069_linker','duplicates':'k_e069_duplicates','linker_eval':'k_e069_linker_cpu','image_cache':'k_e069_images','local_linker_eval':'k_e069_local_cpu'}[mode]
    slug={'capture':'biohub-e069-linker-cache','duplicates':'biohub-e069-duplicates-cpu','linker_eval':'biohub-e069-linker-cpu','image_cache':'biohub-e069-images-cpu','local_linker_eval':'biohub-e069-local-linker-cpu'}[mode]
    nb=build.clean(build.SRC);cells=nb['cells']
    filter_src,official,_=build._lab_embed_common()
    text=build.src(cells[5])
    linefit=next(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='linefit_smooth_output_graph')
    base=(W/'cpu_lab7.py').read_text(encoding='utf-8').split('# Independently compare batched replay')[0]
    base=base.replace('OUT = WORKING_DIR / "cpu7"',f'OUT = WORKING_DIR / "e069_{mode}"')
    replacements={'__OFFICIAL_FILES__':repr(official),'__FILTER_SRC__':repr(filter_src),'__LINEFIT_SRC__':repr(linefit),
                  '__REPLAY_SRC__':repr((W/'replay_linefit.py').read_text()),'__XR_SRC__':repr((W/'extra_rules.py').read_text()),
                  '__C3_VARIANTS_SRC__':repr((W/'c3_variants.py').read_text())}
    for marker,value in replacements.items(): base=base.replace(marker,value)
    runner=(W/'e069_runner.py').read_text(encoding='utf-8')
    for marker,file in [('__INDEPENDENT_RULES__','independent_rules.py'),('__INDEPENDENT_LINKER__','independent_linker.py'),('__MODEL_DEFINITIONS__','hengck_model_v12.py')]:
        runner=runner.replace(marker,repr((W/file).read_text(encoding='utf-8')))
    checkpoint=build.W.parents[1]/'outputs/e069/weights/00000008.pth'
    checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload=base+'\n'+runner
    manifest=dict(experiment='E069',mode=mode,payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),checkpoint_sha256=checkpoint_sha,
                  baseline='C3_public0955',panel_seed='e069-fixed-panel-1',internet=False)
    capture={}
    if mode=='linker_eval': capture=json.loads((W/'k_e069_linker/expected_manifest.json').read_text())
    if mode=='local_linker_eval': capture=json.loads((W/'k_e069_images/expected_manifest.json').read_text())
    expected_rows={}
    if mode in ('local_linker_eval','linker_eval'):
        with (W.parents[1]/'results/E068/cpu7_rows.csv').open(newline='') as f:
            for r in csv.DictReader(f):
                if r['config']=='C3_public0955':
                    expected_rows[r['stem']]={k:float(r[k]) for k in ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes','adj_edge_jaccard')}
        assert len(expected_rows)==199
    header=f'MODE={mode!r}\nRUN_MANIFEST={manifest!r}\nEXPECTED_CHECKPOINT_SHA={checkpoint_sha!r}\nCAPTURE_MANIFEST={capture!r}\nEXPECTED_BASELINE={expected_rows!r}\n'
    ast.parse(header+payload)
    pre=copy.deepcopy(cells[8]);build.set_src(pre,'import json\nimport math\nfrom pathlib import Path\nimport numpy as np\nVOXEL_SCALE_UM=(1.625,.40625,.40625)\n')
    lab=copy.deepcopy(cells[8]);build.set_src(lab,header+payload)
    nb['cells']=[cells[0],cells[2],cells[3],pre,lab]
    sources=SOURCES+(['jarturo/biohub-e069-linker-cache'] if mode=='linker_eval' else [])
    datasets=list(build.DATASETS)+(['hengck23/hengck23-cell-point-detector-demo'] if mode=='capture' else [])
    if mode=='local_linker_eval': datasets.append('jarturo/biohub-e069-local-linker-cache')
    build.write_kernel(folder,slug,slug.replace('-',' '),nb,datasets=datasets,kernel_sources=sources,gpu=mode=='capture')
    (W/folder/'expected_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['capture','duplicates','linker_eval','image_cache','local_linker_eval']);make(p.parse_args().mode)
