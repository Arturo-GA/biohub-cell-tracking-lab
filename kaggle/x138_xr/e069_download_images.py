"""Download the immutable CPU image export with four parallel read requests."""
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import requests
from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'outputs/e069/image_fast'
REF='jarturo/biohub-e069-images-cpu'

def valid(path):
    if not path.exists() or not path.stat().st_size:return False
    try:
        if path.suffix=='.npz':
            with zipfile.ZipFile(path) as z:return z.testzip() is None
        if path.suffix=='.gz':
            with gzip.open(path,'rt',encoding='utf-8') as f:json.load(f)
        else:json.loads(path.read_text())
        return True
    except (OSError,ValueError,EOFError,zipfile.BadZipFile):return False

def main():
    api=KaggleApi();api.authenticate();assert api.kernels_status(REF).status.name=='COMPLETE'
    with api.build_kaggle_client() as client:
        q=ApiListKernelSessionOutputRequest();q.user_name='jarturo';q.kernel_slug=REF.split('/')[1];q.page_size=100
        response=client.kernels.kernels_api_client.list_kernel_session_output(q)
        assert not response.next_page_token
    files=[x for x in response.files if x.file_name.startswith('e069_image_cache/')]
    # Small metadata first, then image bundles in alphabetical order.
    files.sort(key=lambda x:('bundles/' in x.file_name,'candidates/' in x.file_name,x.file_name))
    def fetch(item):
        dest=(DEST/item.file_name).resolve()
        assert dest.is_relative_to(DEST.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True)
        if valid(dest):return item.file_name,'cached'
        for origin in ('image_cache','image_meta'):
            old=ROOT/'outputs/e069'/origin/item.file_name
            if valid(old):shutil.copyfile(old,dest);return item.file_name,'reused'
        tmp=dest.with_suffix(dest.suffix+'.part')
        for attempt in range(3):
            try:
                with requests.get(item.url,stream=True,timeout=(30,180)) as r:
                    r.raise_for_status()
                    with tmp.open('wb') as f:
                        for chunk in r.iter_content(1024*1024):f.write(chunk)
                tmp.replace(dest)
                assert valid(dest),item.file_name
                return item.file_name,'downloaded'
            except requests.RequestException:
                if attempt==2:raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(fetch,x) for x in files]
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            name,status=future.result();print(i,'/',len(files),status,name,flush=True)
    source=DEST/'e069_image_cache'
    expected=json.loads((ROOT/'kaggle/x138_xr/k_e069_images/expected_manifest.json').read_text())
    assert json.loads((source/'run_manifest.json').read_text())==expected
    panel=json.loads((source/'panel.json').read_text());assert len(panel)==32
    launch=json.loads((ROOT/'results/E069/image_cache_launch.json').read_text())
    assert hashlib.sha256((ROOT/'kaggle/x138_xr/k_e069_images/notebook.ipynb').read_bytes()).hexdigest()==launch['notebook_sha256']
    result=dict(kernel=REF,version=launch['version'],files=len(files),summary=json.loads((source/'complete.json').read_text()),manifest_verified=True,folder=str(source))
    (ROOT/'results/E069/image_cache_completed.json').write_text(json.dumps(result,indent=2)+'\n')
    print('VERIFIED',len(panel),'image bundles',flush=True)

if __name__=='__main__':main()
