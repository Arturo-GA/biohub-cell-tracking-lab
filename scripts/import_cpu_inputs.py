"""Verify and import a CPU-prepared image shard into the bounded local cache."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from biohub_lab.event_portable import sha, signature


def import_package(source, destination):
    source=Path(source);destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'image_package.json').read_text());archive=source/manifest['archive']
    if not archive.resolve().is_relative_to(source.resolve()) or sha(archive)!=manifest['archive_sha256']:
        raise ValueError('Image archive path/checksum mismatch')
    if manifest['annotations_read'] or signature(manifest['image_files'])!=manifest['image_sha256']:
        raise ValueError('Image-only package manifest invalid')
    if manifest['source_bytes']>6*2**30:
        raise ValueError('Image exceeds local cache budget')
    with zipfile.ZipFile(archive) as package:
        files=[item for item in package.infolist() if not item.is_dir()]
        expected={manifest['video']+'.zarr/'+name:digest for name,digest in manifest['image_files'].items()}
        if len(files)!=len(expected) or {item.filename for item in files}!=set(expected):
            raise ValueError('Archive contents differ from the image manifest')
        required=0
        for item in files:
            target=(destination/item.filename).resolve()
            if not target.is_relative_to(destination):raise ValueError('Unsafe archive path')
            if target.exists():
                if sha(target)!=expected[item.filename]:raise ValueError('Existing local image differs')
            else:required+=item.file_size
        cached=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
        if cached+required>6*2**30 or shutil.disk_usage(destination).free-required<10*2**30:
            raise ValueError('Insufficient local cache/disk reserve; preserve existing files and rotate a verified completed shard first')
        for item in files:
            target=destination/item.filename
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_name(target.name+'.part')
                with package.open(item) as inp,temp.open('wb') as out:shutil.copyfileobj(inp,out)
                if sha(temp)!=expected[item.filename]:raise ValueError('Image chunk checksum mismatch')
                temp.replace(target)
    return dict(video=manifest['video'],image=str(destination/(manifest['video']+'.zarr')),verified_files=len(expected))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--destination',required=True)
    args=parser.parse_args();print(json.dumps(import_package(args.source,args.destination),indent=2))
