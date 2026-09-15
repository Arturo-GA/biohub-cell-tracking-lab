"""CPU image/gaussian preparation for the next local video; no GEFF access."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

import zarr
from biohub_lab.event_data import save_json
from biohub_lab.event_portable import sha, signature
from biohub_lab.gaussian_detector import GAUSSIAN_CONFIG, run_video
from notebook_runner import competition_dir


def main(package, offset=0, root=None):
    package=Path(package);root=Path(root or '/kaggle/working/local_inputs');root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((package/'baseline/event_graph_split.json').read_text());split=manifest['split']
    data=competition_dir()/'train';inventory=sorted(p.stem for p in data.glob('*.zarr'))
    if signature(inventory)!=manifest['training_names_sha256']:
        # The original inventory hash uses compact JSON and alphabetical names.
        expected=hashlib.sha256(json.dumps(inventory,separators=(',',':')).encode()).hexdigest()
        if expected!=manifest['training_names_sha256']:raise ValueError('Competition inventory changed')
    ordered=split['fit']+split['calibration'];name=ordered[offset]
    image=data/(name+'.zarr');files=sorted(p for p in image.rglob('*') if p.is_file())
    size=sum(p.stat().st_size for p in files)
    if size>6*2**30:raise ValueError('Video exceeds initial local disk cache budget; enlarge the explicit budget before export')
    source_hashes={p.relative_to(image).as_posix():sha(p) for p in files}
    started=time.monotonic();archive=root/(name+'.zip')
    # Zarr chunks are already compressed. Storing them avoids unnecessary CPU work.
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED,allowZip64=True) as output:
        for p in files:output.write(p,name+'.zarr/'+p.relative_to(image).as_posix())
    save_json(root/'image_package.json',dict(video=name,offset=offset,next_offset=offset+1,
        archive=archive.name,archive_sha256=sha(archive),source_bytes=size,image_sha256=signature(source_hashes),
        image_files=source_hashes,shape=list(map(int,zarr.open_group(str(image),mode='r')['0'].shape)),
        annotations_read=False,split_manifest_sha256=sha(package/'baseline/event_graph_split.json')))
    print('CPU_IMAGE_PACKAGE_READY',name,size,flush=True)
    proposals=root/'gaussian.npz';run_video(image,proposals)
    save_json(root/'gaussian_receipt.json',dict(video=name,image_sha256=signature(source_hashes),
        proposals_sha256=sha(proposals),config=GAUSSIAN_CONFIG,annotations_read=False))
    save_json(root/'result.json',dict(status='complete',video=name,offset=offset,next_offset=offset+1,
        archive_sha256=sha(archive),gaussian_sha256=sha(proposals),seconds=time.monotonic()-started,
        annotations_read=False,accelerator='none',training_started=False,
        remaining_input_videos=ordered[offset+1:],scope='One resumable input shard of the fixed 64 fit/calibration videos, not a reduced training experiment'))


def select_batch(ordered,offset,count,sizes,budget=4*2**30):
    """Return a contiguous bounded prefix; never skip a video to fit the budget."""
    if not 0<=offset<len(ordered) or not 1<=count<=4 or offset+count>len(ordered):
        raise ValueError('Batch must contain 1-4 videos within the fixed input cohort')
    selected=[];total=0
    for name in ordered[offset:offset+count]:
        size=sizes[name]
        if size<=0:raise ValueError('Missing or empty image store: '+name)
        if total+size>budget:break
        selected.append(name);total+=size
    if not selected:raise ValueError('First video exceeds the batch budget; do not skip it')
    return selected,total


def batch(package,offset,count):
    package=Path(package);root=Path('/kaggle/working/local_inputs');root.mkdir(exist_ok=True)
    split=json.loads((package/'baseline/event_graph_split.json').read_text())['split']
    ordered=split['fit']+split['calibration'];data=competition_dir()/'train'
    sizes={name:sum(p.stat().st_size for p in (data/(name+'.zarr')).rglob('*') if p.is_file())
           for name in ordered[offset:offset+count]}
    names,total=select_batch(ordered,offset,count,sizes)
    if shutil.disk_usage(root).free-total<2*2**30:raise ValueError('Insufficient CPU output disk reserve')
    record=dict(status='preparing',offset=offset,requested_count=count,videos=names,completed=[],
        next_offset=offset+len(names),source_bytes=total,budget_bytes=4*2**30,
        annotations_read=False,accelerator='none',training_started=False,
        scope='Contiguous input batch; the full fixed 48/16/48 experiment is unchanged')
    save_json(root/'batch_result.json',record)
    try:
        for index,name in enumerate(names,offset):
            main(package,index,root/name);record['completed'].append(name)
            save_json(root/'batch_result.json',record)
        record['status']='complete';save_json(root/'batch_result.json',record)
    except Exception as error:
        record.update(status='failed',error=str(error));save_json(root/'batch_result.json',record);raise


if __name__=='__main__':
    import sys
    offset=int(sys.argv[2]) if len(sys.argv)>2 else 0
    count=int(sys.argv[3]) if len(sys.argv)>3 else 1
    if count==1:main(sys.argv[1],offset)
    else:batch(sys.argv[1],offset,count)
