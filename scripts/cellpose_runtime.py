"""Install exactly the CPU-prepared wheels in an offline Kaggle runtime."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
def locate(suffix):
    found=list(Path('/kaggle/input').rglob(suffix));assert len(found)==1,(suffix,len(found));return found[0]
def setup(package):
    path=locate('cellpose_assets/result.json');assets=json.loads(path.read_text())
    pin=json.loads((package/'baseline/e028_runtime.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pin['manifest_sha256'] and assets['status']=='complete'
    wheels=[]
    for item in assets['wheels']:
        wheel=path.parent/'wheels'/item['file'];assert hashlib.sha256(wheel.read_bytes()).hexdigest()==item['sha256'];wheels.append(str(wheel))
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*wheels],check=True)
    os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD']='1'
    return path.parent,assets
