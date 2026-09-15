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


def main(package, offset=0):
    package=Path(package);root=Path('/kaggle/working/local_inputs');root.mkdir(exist_ok=True)
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


if __name__=='__main__':
    import sys
    main(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 0)
