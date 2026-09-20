"""E041: download licensed NIS3D and prepare dense targets on CPU."""
import hashlib,json,sys,time,urllib.request,zipfile,shutil
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import tifffile

def main(package,archive_source=None):
    start=time.monotonic();out=Path('/kaggle/working/dense_supervision');out.mkdir(exist_ok=True)
    url='https://zenodo.org/api/records/11456029/files/NIS3D.zip/content'
    archive=out/'NIS3D.zip' if archive_source is None else archive_source;digest=hashlib.md5();total=0;last=time.monotonic();reuse=None
    if archive_source is None:
        with urllib.request.urlopen(url,timeout=120) as response,archive.open('wb') as f:
            while chunk:=response.read(4*1024*1024):
                f.write(chunk);digest.update(chunk);total+=len(chunk)
                if time.monotonic()-last>30:print('DENSE_DOWNLOAD',total,flush=True);last=time.monotonic()
    else:
        with archive.open('rb') as f:
            while chunk:=f.read(4*1024*1024):digest.update(chunk);total+=len(chunk)
        # Rebuild all targets with the exact interpolation grid, not nominal zoom.
    assert total==3296003909 and digest.hexdigest()=='f229013526645c79d31bc58ba7f12be7'
    records=[];rng=np.random.default_rng(410920)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();print('DENSE_ARCHIVE',json.dumps(names),flush=True)
        for name in sorted(n for n in names if n.startswith('NIS3D/NIS3D/') and n.lower().endswith('/data.tif')):
            prefix=name[:-len('Data.tif')]
            if reuse and '/Zebrafish_1/' not in name:
                old=next(r for r in reuse['videos'] if r['source']==name);path=archive.parent/old['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==old['sha256'];shutil.copy2(path,out/path.name)
                records.append(dict(old,voxel_spacing_um=[1.,1.,1.],reused_isotropic=True));print('DENSE_REUSE',name,flush=True);continue
            def read(n):
                aliases={'Data.tif':['data.tif'],'GroundTruth.tif':['gt.tif'],'ConfidenceScore.tif':['scoreOfConfidence.tif']}
                actual=next(prefix+a for a in [n]+aliases.get(n,[]) if prefix+a in names)
                with z.open(actual) as f:return tifffile.imread(f)
            image=read('Data.tif');labels=read('GroundTruth.tif');confidence=read('ConfidenceScore.tif')
            assert image.ndim==3 and image.shape==labels.shape==confidence.shape
            ids,counts=np.unique(labels,return_counts=True);ids=ids[ids>0]
            centers=np.asarray(ndi.center_of_mass(np.ones(labels.shape,np.uint8),labels,ids))
            cell_conf=ndi.maximum(confidence,labels,ids);centers=centers[cell_conf>=3]
            goodcounts=ndi.sum(np.ones(labels.shape,np.uint8),labels,ids)[cell_conf>=3]
            # Equalize nuclear size across external acquisition domains; no Biohub GT used.
            spacing=np.array([2.5,.43,.43]) if '/Zebrafish_1/' in name else np.ones(3)
            radius=float(np.median((3*goodcounts*np.prod(spacing)/(4*np.pi))**(1/3)));factor=np.clip(2.5/radius,.15,2.)*spacing
            lo,hi=np.quantile(image,[.001,.999]);image=ndi.zoom(np.clip((image.astype(np.float32)-lo)/max(float(hi-lo),1e-6),0,1),factor,order=1)
            valid=ndi.zoom((confidence!=1)&(confidence!=2),factor,order=0)
            # ndimage.zoom(grid_mode=False) aligns endpoints. Its rounded output
            # shape means multiplying points by nominal zoom introduces drift.
            coordinate_scale=(np.asarray(image.shape)-1)/(np.asarray(labels.shape)-1)
            centers*=coordinate_scale
            # Ignore uncertain regions with a margin; do not turn them into background.
            valid=~ndi.binary_dilation(~valid,iterations=3)
            shape=np.array(image.shape);xs=[];ys=[];ms=[]
            for j in range(512):
                c=centers[rng.integers(len(centers))]+rng.uniform(-10,10,3) if j%2 else rng.uniform(0,shape)
                low=np.rint(c).astype(int)-16;grid=np.indices((32,32,32)).transpose(1,2,3,0)+low
                inside=np.all((grid>=0)&(grid<shape),axis=-1);clipped=np.clip(grid,0,shape-1);ix=tuple(clipped[...,a] for a in range(3))
                target=np.zeros((32,32,32),np.float32)
                near=centers[np.all((centers>=low-5)&(centers<low+37),axis=1)]
                for p in near:target=np.maximum(target,np.exp(-np.sum((grid-p)**2,axis=-1)/(2*1.2**2)))
                xs.append(image[ix].astype(np.float16));ys.append(target.astype(np.float16));ms.append(valid[ix]&inside)
            path=out/f'volume_{len(records)}.npz';np.savez_compressed(path,x=xs,y=ys,mask=ms)
            info=z.read(prefix+'Info.txt').decode('utf8',errors='replace') if prefix+'Info.txt' in names else None
            rec=dict(source=name,info=info,file=path.name,centers=len(centers),rescale=factor.tolist(),coordinate_scale=coordinate_scale.tolist(),voxel_spacing_um=spacing.tolist(),shape=list(map(int,shape)),patches=512,sha256=hashlib.sha256(path.read_bytes()).hexdigest());records.append(rec);print('DENSE_PREP',json.dumps(rec),flush=True)
    assert records
    (out/'result.json').write_text(json.dumps(dict(status='complete',videos=records,seconds=time.monotonic()-start,license='CC-BY-4.0',source=url,source_md5=digest.hexdigest(),scope='External dense training; confidence 1 and 2 ignored with margin'),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
