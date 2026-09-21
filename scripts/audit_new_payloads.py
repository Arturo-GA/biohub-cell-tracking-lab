"""Validate embedded bytes and module-level local-package imports.

Lazy function imports are not executed or proven by this static check.
"""
import ast,base64,hashlib,io,json,sys,zipfile
from pathlib import Path

def check(folder):
    folder=Path(folder);meta=json.loads((folder/'kernel-metadata.json').read_text());manifest=json.loads((folder/'payload.json').read_text());nb=json.loads((folder/'notebook.ipynb').read_text())
    assert meta['is_private'] and not meta['enable_internet'] and not meta['enable_tpu']
    code=''.join(nb['cells'][1]['source']);tree=ast.parse(code)
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
    payload=base64.b64decode(ast.literal_eval(node.value.args[0]));assert hashlib.sha256(payload).hexdigest()==manifest['sha256']
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        files=set(z.namelist());assert files==set(manifest['files'])
        for filename in files:
            if not filename.endswith('.py'):continue
            source=z.read(filename);compile(source,filename,'exec')
            for n in ast.parse(source).body:
                if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('biohub_lab.'):
                    target='src/'+n.module.replace('.','/')+'.py';assert target in files,(filename,target)
                elif isinstance(n,ast.ImportFrom) and n.level==1 and filename.startswith('src/biohub_lab/') and n.module:
                    target='src/biohub_lab/'+n.module.replace('.','/')+'.py';assert target in files,(filename,target)
    return dict(kernel=meta['id'],files=len(files),payload_sha256=manifest['sha256'],compiled=True,module_level_local_imports_closed=True)
if __name__=='__main__':
    results=[check(folder) for folder in sys.argv[1:]];print(json.dumps(results,indent=2))
