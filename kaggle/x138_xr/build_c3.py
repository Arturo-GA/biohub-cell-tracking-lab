"""Build fixed c3 candidates from the same parameters as the CPU7 lab."""
import argparse
import ast
import hashlib
import json

import build
from c3_variants import C3_VARIANTS, xr_environment


def make(variant):
    slug = 'biohub-' + variant.lower().replace('_', '-')
    folder = 'k_' + variant.lower()
    overrides = {'BIOHUB_OUTPUT_MIN_TRACK_LEN': '7', 'BIOHUB_REPAIR_DEADLINE_S': '34200'}
    xr = xr_environment(variant)
    build.make_xr(folder, slug, 'Biohub ' + variant.lower().replace('_', ' '), overrides, xr)
    path = build.W / folder / 'notebook.ipynb'
    nb = json.loads(path.read_text(encoding='utf-8'))
    code = '\n'.join(build.src(c) for c in nb['cells'] if c['cell_type'] == 'code')
    manifest = dict(candidate=variant, base='c3_public_0.955',
                    source_sha256=hashlib.sha256(code.encode()).hexdigest(), fixed=overrides, xr=xr)
    marker = ('import json as _manifest_json\nfrom pathlib import Path as _ManifestPath\n'
              '_manifest = ' + repr(manifest) + '\n'
              '_ManifestPath("/kaggle/working/run_manifest.json").write_text(_manifest_json.dumps(_manifest, sort_keys=True))\n'
              'print("RUN_MANIFEST", _manifest_json.dumps(_manifest, sort_keys=True))\n')
    nb['cells'].append(dict(cell_type='code', metadata={}, outputs=[], execution_count=None,
                            source=marker.splitlines(keepends=True)))
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            ast.parse(build.src(c))
    path.write_text(json.dumps(nb, indent=1), encoding='utf-8')
    (path.parent / 'expected_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    meta_path = path.parent / 'kernel-metadata.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    meta['machine_shape'] = 'NvidiaTeslaT4'
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('PREPARED', variant, manifest['source_sha256'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('variant', choices=list(C3_VARIANTS))
    make(p.parse_args().variant)
