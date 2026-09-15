"""Verify both CPU payloads without starting a remote notebook."""
import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from launch_cpu_notebook import validate

ROOT=Path(__file__).resolve().parents[1]


def main():
    reports={}
    for name in ('cpu_control','cpu_prepare'):
        folder=ROOT/'kaggle'/name;meta=validate(folder)
        notebook=json.loads((folder/'notebook.ipynb').read_text());tree=ast.parse(''.join(notebook['cells'][1]['source']))
        assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
        payload=base64.b64decode(ast.literal_eval(assignment.value.args[0]));receipt=json.loads((folder/'payload.json').read_text())
        if hashlib.sha256(payload).hexdigest()!=receipt['sha256']:raise ValueError('Payload checksum differs')
        target=ROOT/'artifacts/cpu_payload_verify'/name;target.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for path in archive.namelist():
                if not (target/path).resolve().is_relative_to(target.resolve()):raise ValueError('Unsafe archive path')
                if path.endswith('.py'):ast.parse(archive.read(path).decode())
            archive.extractall(target)
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(target/'src'),str(target/'scripts')]),CUDA_VISIBLE_DEVICES='')
        subprocess.run([sys.executable,'-c','import kaggle_cpu_control, kaggle_cpu_prepare; print("Minimal CPU package imports pass")'],
                       env=env,cwd=target,check=True)
        reports[name]=dict(kernel=meta['id'],payload_sha256=receipt['sha256'],private_cpu_metadata_verified=True,
            notebook_and_packaged_python_compile=True,minimal_payload_imports=True,
            offline_kaggle_dependency_install_not_executed_locally=True)
    (ROOT/'results/CPU_WORKFLOW_PREFLIGHT.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
