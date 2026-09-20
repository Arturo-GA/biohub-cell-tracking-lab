"""Download only small E027 reference/center arrays for a local CPU diagnosis."""
import hashlib,json
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest
ROOT=Path(__file__).resolve().parents[1]
def main():
    assert json.loads((ROOT/'results/E027_completed.json').read_text())['status']=='complete'
    frames=json.loads((ROOT/'outputs/e027_inputs_recovery/nucverse_inputs/result.json').read_text())['frames']
    stems={Path(r['file']).stem for r in frames};out=ROOT/'outputs/e027_diagnostic_data';out.mkdir(exist_ok=True)
    api=KaggleApi();api.authenticate();downloaded=[]
    for slug,prefix,suffix in [('biohub-lab-error-window-inputs-cpu','nucverse_inputs/','_reference.npz'),('biohub-lab-error-window-evaluation-cpu','nucverse_evaluation/','_centers.npy')]:
        expected={prefix+stem+suffix for stem in stems};urls={};token=''
        with api.build_kaggle_client() as client:
            while True:
                req=ApiListKernelSessionOutputRequest();req.user_name='jarturo';req.kernel_slug=slug;api._set_paging(req,100,token)
                response=client.kernels.kernels_api_client.list_kernel_session_output(req)
                for item in response.files or []:
                    if item.file_name in expected:urls[item.file_name]=item.url
                token=response.next_page_token
                if not token:break
        assert set(urls)==expected
        for name,url in sorted(urls.items()):
            r=requests.get(url,timeout=(20,60));r.raise_for_status();assert len(r.content)<10_000_000
            target=out/Path(name).name;assert target.resolve().is_relative_to(out.resolve());target.write_bytes(r.content)
            downloaded.append(dict(file=target.name,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest()))
    (out/'download_manifest.json').write_text(json.dumps(downloaded,indent=2)+'\n');print(json.dumps(dict(files=len(downloaded),bytes=sum(r['bytes'] for r in downloaded))))
if __name__=='__main__':main()
