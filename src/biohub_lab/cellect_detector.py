"""CELLECT pretrained two-frame UNet, adapted to Biohub ZYX coordinates.

Uses the original localization class 4 / foreground-neighborhood extraction.
Physical NMS and temporal corroboration replace CELLECT's intra-frame MLP.
This is a detector adapter, not a reproduction of the full CELLECT tracker.
"""
import hashlib
import itertools
from pathlib import Path
import numpy as np
from .detector_proposals import nms, temporal_filter, save_proposals

WEIGHT_SHA = 'f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5'
# Canonical LF hash; upstream distributes this source with CRLF line endings.
MODEL_SHA = 'bb0902db6161fa8e8fb4eac5ff58d0b6cc4bcf0754cfaecdbc5128f8dc9724da'
CELLECT_CONFIG = dict(tile_yxz=[256, 256, 32], overlap_yxz=[16, 16, 8],
                     preprocessing='log1p(max(raw, positive_min) + 1900)',
                     localization_probability=.5, nms_um=3., last_frame='repeat_current')


def tile_starts(length, size, overlap):
    if length <= size:
        return [0]
    return sorted(set(list(range(0, length - size + 1, size - overlap)) + [length - size]))


def preprocess(pair):
    pair = np.clip(np.asarray(pair, np.float32), 0, 65535)
    positive = pair[pair > 0]
    if len(positive):
        pair = np.maximum(pair, positive.min())
    return np.log1p(pair + 1900.)


def decode(seg, loc):
    import torch
    import torch.nn.functional as F
    kernel = torch.tensor([[0,0,0,1,0,0,0], [0,0,1,2,1,0,0], [0,1,2,3,2,1,0],
                           [1,2,3,4,3,2,1], [0,1,2,3,2,1,0], [0,0,1,2,1,0,0],
                           [0,0,0,1,0,0,0]], device=loc.device, dtype=torch.float32)
    kernel = kernel.reshape(1, 1, 7, 7, 1) / kernel.sum()
    smooth = F.conv3d(loc.float(), kernel.expand(5, 1, 7, 7, 1), padding=(3,3,0), groups=5)
    maximum = F.max_pool3d(smooth.max(1, keepdim=True).values, 3, 1, 1)[:, 0]
    background = (seg.argmax(1, keepdim=True) == 0).float()
    neighborhood = torch.zeros((1,1,3,3,3), device=loc.device)
    for a,b,c in itertools.product((-1,0,1), repeat=3):
        if abs(a) + max(abs(b),abs(c)) <= 1:
            neighborhood[0,0,a+1,b+1,c+1] = 1
    foreground = F.conv3d(background, neighborhood, padding=1)[:,0] == 0
    probability = smooth.softmax(1)[:, 4]
    peak = (smooth[:,4] == maximum) & foreground & (probability >= .5)
    peak[:,:2] = False; peak[:,-2:] = False
    peak[:,:,:2] = False; peak[:,:,-2:] = False
    peak[:,:,:,:1] = False; peak[:,:,:,-1:] = False
    coords = peak.nonzero()
    scores = probability[tuple(coords.T)].cpu().numpy()
    return coords[:,1:].cpu().numpy(), scores


def detect_pair(model, pair, device):
    import torch
    # Original CELLECT uses YXZ axes, with Z retained at every UNet level.
    volume = preprocess(pair).transpose(0, 2, 3, 1)
    shape = volume.shape[1:]
    tile = CELLECT_CONFIG['tile_yxz']
    padding = [(0,0)] + [(0,max(0,s-n)) for n,s in zip(shape,tile)]
    volume = np.pad(volume, padding, mode='edge')
    coords, scores = [], []
    starts = [tile_starts(n,s,o) for n,s,o in zip(volume.shape[1:],tile,CELLECT_CONFIG['overlap_yxz'])]
    with torch.inference_mode():
        for offset in itertools.product(*starts):
            slices = tuple(slice(v,v+s) for v,s in zip(offset,tile))
            x = torch.from_numpy(np.ascontiguousarray(volume[(slice(None),)+slices])).unsqueeze(0).to(device)
            with torch.autocast('cuda', enabled=str(device).startswith('cuda'), dtype=torch.float16):
                seg, loc, features, division, size = model(x)
            p, s = decode(seg, loc)
            p = p + np.array(offset)
            valid = np.all(p < np.array(shape), axis=1)
            coords.append(p[valid][:,[2,0,1]])
            scores.append(s[valid])
            del x, seg, loc, features, division, size
    coords, scores = np.concatenate(coords), np.concatenate(scores)
    keep = nms(coords, scores)
    return coords[keep].astype(np.float32), scores[keep]


def run_video(image_path, output, weights, device='cuda:0'):
    import torch
    import zarr
    from biohub_cellect import model as architecture
    assert hashlib.sha256(Path(architecture.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == MODEL_SHA
    assert hashlib.sha256(Path(weights).read_bytes()).hexdigest() == WEIGHT_SHA
    model = architecture.UNet3D(2,6).to(device).eval()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    image = zarr.open_group(str(image_path), mode='r')['0']
    frames = []
    first = np.asarray(image[0])
    for t in range(image.shape[0]):
        second = np.asarray(image[min(t+1,image.shape[0]-1)])
        frames.append(detect_pair(model, np.stack([first,second]), device))
        first = second
        if t % 10 == 0:
            print('CELLECT', Path(image_path).stem, t, len(frames[-1][0]), flush=True)
    raw_count = sum(len(p) for p,_ in frames)
    frames = temporal_filter(frames)
    return save_proposals(output, frames, image.shape,
        dict(method='CELLECT native two-frame detector', detector_config=CELLECT_CONFIG,
             raw_proposals=raw_count, weight_sha256=WEIGHT_SHA, architecture_sha256=MODEL_SHA))
