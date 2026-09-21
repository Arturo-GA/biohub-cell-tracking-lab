import hashlib
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def one(relative):
    found=list(dict.fromkeys(p for depth in (1,2,3) for p in Path('/kaggle/input').glob('*/'*depth+relative)))
    assert len(found)==1,(relative,found)
    return found[0]
