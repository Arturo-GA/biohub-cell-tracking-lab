"""Download only the public visible input volumes for local GPU preparation."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import threading
import time
import requests
from kaggle.api.kaggle_api_extended import KaggleApi

# Some CDN connections otherwise hang for many minutes and occupy all workers.
# Every operation in this script is read-only; bounded retries preserve completed files.
_send = requests.sessions.Session.send
def _bounded_send(self, request, **kwargs):
    kwargs['timeout'] = (15, 90)
    return _send(self, request, **kwargs)
requests.sessions.Session.send = _bounded_send

def _atomic_download(self, response, outfile, http_client, quiet=True, resume=False, **kwargs):
    # Re-open the SDK's same authorized URL with an explicit requests socket timeout.
    # The initial SDK stream can bypass Session.send's timeout on some versions.
    url=response.url
    headers=dict(response.request.headers)
    method=response.request.method
    expected=int(response.headers['Content-Length']) if response.headers.get('Content-Length') else None
    response.close()
    target=Path(outfile);partial=target.with_name(target.name+'.e071-partial')
    with requests.request(method,url,headers=headers,stream=True,timeout=(15,90)) as stream:
        stream.raise_for_status()
        with partial.open('wb') as f:
            for chunk in stream.iter_content(65536):
                if chunk:f.write(chunk)
    if expected is not None and partial.stat().st_size!=expected:
        raise ValueError('Incomplete download')
    partial.replace(target)
KaggleApi.download_file = _atomic_download

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/e071_runtime/competition'
page = json.loads((ROOT/'results/E071/competition_file_page.json').read_text())
all_rows = list(page['files'])
if page.get('nextPageToken'):
    api = KaggleApi(); api.authenticate()
    while page.get('nextPageToken'):
        page = api.competition_list_files('biohub-cell-tracking-during-development',
            page_token=page['nextPageToken'], page_size=200).to_dict()
        all_rows.extend(page['files'])
        if any(r['name'].startswith('train/') for r in page['files']): break
rows = [r for r in all_rows if r['name'].startswith('test/')]
(ROOT/'results/E071/visible_input_manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
stems = {r['name'].split('/')[1] for r in rows}
assert len(stems)==4 and len(rows)>=400
assert shutil.disk_usage(ROOT).free > sum(r['totalBytes'] for r in rows) + 2*1024**3
local = threading.local()


def one(row):
    name=row['name'];path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size==row['totalBytes']: return name
    if not hasattr(local,'api'):
        local.api=KaggleApi();local.api.authenticate()
    for attempt in range(3):
        try:
            local.api.competition_download_file('biohub-cell-tracking-during-development',name,
                path=str(path.parent),force=True,quiet=True)
            assert path.exists() and path.stat().st_size==row['totalBytes'], (name,'size mismatch')
            return name
        except Exception:
            if attempt==2:raise RuntimeError('Download failed for '+name) from None
            time.sleep(3*(attempt+1))


if __name__=='__main__':
    started=time.monotonic();print('Visible files',len(rows),'bytes',sum(r['totalBytes'] for r in rows),flush=True)
    failed=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        tasks=[pool.submit(one,r) for r in rows]
        for i,f in enumerate(as_completed(tasks),1):
            try:f.result()
            except Exception as exc:failed.append(str(exc));print('Download retry required:',type(exc).__name__,flush=True)
            if i%25==0:print('VISIBLE_DOWNLOAD',i,'/',len(rows),round(time.monotonic()-started,1),'s',flush=True)
    assert not failed, f'{len(failed)} files failed; valid completed files preserved'
    result=dict(files=len(rows),stems=sorted(stems),bytes=sum(r['totalBytes'] for r in rows),seconds=time.monotonic()-started)
    (ROOT/'results/E071/local_visible_complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
