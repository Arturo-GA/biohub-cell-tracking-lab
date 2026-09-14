"""Resolve overlapping nuclei with one-versus-two 3D Gaussian image fits.

All proposal decisions use raw intensities and adjacent images. No labels,
learned association scores, diagnostic event times, or cached detections enter.
"""
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import least_squares
from .detector_proposals import SCALE, nms, temporal_filter, save_proposals

GRID = SCALE * np.array([1, 2, 2])
GAUSSIAN_CONFIG = dict(downsample=[1,2,2], seed_sigma_um=1.2, seed_nms_um=4.,
    radius_um=6.5, max_fits_per_frame=48, min_elongation=1.35,
    min_bic_gain=10., min_residual_reduction=.2, min_separation_um=2.6,
    max_separation_um=10., min_amplitude_ratio=.25, max_nfev=40)


def model_and_jac(params, xyz, components):
    """Shared anisotropic width, separate amplitudes/centers, constant background."""
    background, sigma = params[0], params[1:4]
    value = np.full(len(xyz), background)
    jac = np.zeros((len(xyz), len(params)))
    jac[:, 0] = 1
    for k in range(components):
        start = 4 + 4*k
        amplitude, center = params[start], params[start+1:start+4]
        delta = xyz - center
        kernel = np.exp(-.5*np.sum((delta/sigma)**2, axis=1))
        g = amplitude * kernel
        value += g
        jac[:, 1:4] += g[:,None] * delta**2 / sigma**3
        jac[:, start] = kernel
        jac[:, start+1:start+4] = g[:,None] * delta / sigma**2
    return value, jac


def moments(patch, xyz):
    weights = np.maximum(patch.ravel() - np.quantile(patch, .2), 0)
    weights /= max(weights.sum(), 1.e-12)
    center = np.sum(xyz * weights[:,None], axis=0)
    delta = xyz - center
    covariance = (delta*weights[:,None]).T @ delta
    return center, covariance


def fit_split(patch, spacing=GRID):
    """Return two physical offsets only when a second component explains signal."""
    axes = [(np.arange(n)-(n-1)/2)*s for n,s in zip(patch.shape,spacing)]
    xyz = np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    target = np.asarray(patch, np.float64).ravel()
    if np.ptp(target) < .05:
        return None
    center, cov = moments(patch, xyz)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    elongation = eigenvalues[-1] / max(eigenvalues[-2], 1.e-9)
    if elongation < GAUSSIAN_CONFIG['min_elongation']:
        return None
    sigma = np.clip(np.sqrt(np.diag(cov)), .8, 3.)
    center = np.clip(center, -3., 3.)
    amplitude = np.clip(target.max() - np.quantile(target,.2), .06, 1.5)
    background = float(np.clip(np.quantile(target,.2), 0, .4))
    p1 = np.r_[background,sigma,amplitude,center]
    direction = eigenvectors[:,-1] * min(2.5, np.sqrt(eigenvalues[-1])*.8)
    p2 = np.r_[background,np.clip(sigma*.75,.8,3.),amplitude,center-direction,
               amplitude,center+direction]
    fits = []
    for count, initial in ((1,p1),(2,p2)):
        lower = np.r_[-.1,[.7]*3,np.tile([.02,-5.5,-5.5,-5.5],count)]
        upper = np.r_[.5,[4.]*3,np.tile([2.,5.5,5.5,5.5],count)]
        initial = np.clip(initial,lower+1.e-5,upper-1.e-5)
        fit = least_squares(lambda p:model_and_jac(p,xyz,count)[0]-target, initial,
            jac=lambda p:model_and_jac(p,xyz,count)[1], bounds=(lower,upper),
            max_nfev=GAUSSIAN_CONFIG['max_nfev'], ftol=1.e-4, xtol=1.e-4, gtol=1.e-4)
        fits.append(fit)
    one, two = fits
    error1 = np.sum(one.fun**2); error2 = np.sum(two.fun**2)
    if not one.success or not two.success or error1 < 1.e-10:
        return None
    reduction = 1-error2/error1
    bic_gain = len(target)*np.log(error1/max(error2,1.e-12))-4*np.log(len(target))
    centers = np.stack([two.x[5:8],two.x[9:12]])
    separation = np.linalg.norm(centers[0]-centers[1])
    ratio = min(two.x[4],two.x[8])/max(two.x[4],two.x[8])
    if (bic_gain < GAUSSIAN_CONFIG['min_bic_gain'] or reduction < .2 or
            not 2.6 <= separation <= 10. or ratio < .25):
        return None
    return centers, float(bic_gain), float(reduction)


def prepare_frame(raw):
    raw = np.asarray(raw, np.float32)[:,::2,::2]
    low, high = np.quantile(raw,[.01,.999])
    image = np.clip((raw-low)/max(high-low,1.),0,1)
    smooth = gaussian_filter(image,1.2/GRID)
    background = gaussian_filter(image,6./GRID)
    response = smooth-background
    threshold = max(.04, float(np.median(response)+6*np.median(np.abs(response-np.median(response)))))
    peaks = np.argwhere((response == maximum_filter(response,size=3)) & (response > threshold))
    scores = response[tuple(peaks.T)] if len(peaks) else np.empty(0)
    keep = nms(peaks,scores,distance_um=4.,scale=GRID)
    return image, peaks[keep], scores[keep]


def detect_frame(raw):
    image, seeds, seed_scores = prepare_frame(raw)
    radius = np.ceil(6.5/GRID).astype(int)
    shape = tuple(radius*2+1)
    axes = [(np.arange(n)-(n-1)/2)*s for n,s in zip(shape,GRID)]
    xyz = np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    ranked = []
    for seed, strength in zip(seeds,seed_scores):
        # Border patches lack sufficient evidence for a two-component fit.
        if np.any(seed-radius < 0) or np.any(seed+radius >= image.shape):
            continue
        patch = image[tuple(slice(p-r,p+r+1) for p,r in zip(seed,radius))]
        _, cov = moments(patch,xyz)
        eig = np.linalg.eigvalsh(cov)
        elongation = eig[-1]/max(eig[-2],1.e-9)
        if elongation >= GAUSSIAN_CONFIG['min_elongation']:
            ranked.append((float(elongation*strength),seed,patch))
    ranked.sort(key=lambda x:-x[0])
    coords, scores = [], []
    for _, seed, patch in ranked[:GAUSSIAN_CONFIG['max_fits_per_frame']]:
        split = fit_split(patch)
        if split is None:
            continue
        centers, bic, reduction = split
        native = (seed*GRID + centers)/SCALE
        coords.extend(native)
        scores.extend([reduction, reduction])
    coords = np.asarray(coords,np.float32).reshape(-1,3)
    scores = np.asarray(scores,np.float32)
    keep = nms(coords,scores)
    return (coords[keep],scores[keep]), seeds*np.array([1,2,2]), dict(
        eligible_fits=len(ranked), attempted=min(len(ranked),48), accepted_pairs=len(coords)//2)


def run_video(image_path, output):
    import zarr
    image = zarr.open_group(str(image_path), mode='r')['0']
    frames, supports, statistics = [], [], []
    for t in range(image.shape[0]):
        result, peaks, stats = detect_frame(np.asarray(image[t]))
        frames.append(result)
        supports.append(np.concatenate([peaks,result[0]]))
        statistics.append(stats)
        if t % 10 == 0:
            print('GAUSSIAN',Path(image_path).stem,t,stats,flush=True)
    raw_count = sum(len(p) for p,_ in frames)
    frames = temporal_filter(frames,supports)
    return save_proposals(output,frames,image.shape,dict(method='one-versus-two anisotropic Gaussian fits',
        detector_config=GAUSSIAN_CONFIG, raw_proposals=raw_count, fit_statistics=statistics))
