"""Local spatiotemporal attention with continuation and symmetric division heads."""
import math
import torch
from torch import nn


class LocalAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.geometry = nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, 1))
        self.output = nn.Linear(width, width)
        self.feed = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2*width), nn.SiLU(), nn.Linear(2*width, width))

    def forward(self, h, positions, neighbors, chunk=2048):
        q, k, v = self.qkv(self.norm(h)).chunk(3, dim=-1); pieces = []
        for start in range(0, len(h), chunk):
            ids = neighbors[start:start+chunk]; safe = ids.clamp_min(0)
            delta = positions[safe]-positions[start:start+chunk, None]
            weights = (q[start:start+chunk, None]*k[safe]).sum(-1)/math.sqrt(h.shape[-1])
            weights = weights + self.geometry(delta).squeeze(-1)
            weights = weights.masked_fill(ids < 0, -1e4).softmax(-1)
            pieces.append((weights[..., None]*v[safe]).sum(1))
        h = h + self.output(torch.cat(pieces))
        return h + self.feed(h)


def mlp(inputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.LayerNorm(width),
                         nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))


class EventGraphNet(nn.Module):
    def __init__(self, width=64, layers=4):
        super().__init__(); self.config = dict(width=width, layers=layers)
        # Execution settings do not change checkpoint architecture or weights.
        self.attention_chunk = 2048
        self.checkpoint_activations = False
        self.input = nn.Sequential(nn.LayerNorm(72), nn.Linear(72, width), nn.SiLU())
        self.layers = nn.ModuleList([LocalAttention(width) for _ in range(layers)])
        self.link_head = mlp(4*width+4, width)
        self.division_head = mlp(4*width+10, width)
        self.quality_head = mlp(width, width)

    def encode(self, inputs, positions, neighbors):
        h = self.input(inputs)
        for layer in self.layers:
            if self.checkpoint_activations and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                h = checkpoint(layer, h, positions, neighbors, self.attention_chunk, use_reentrant=False)
            else:
                h = layer(h, positions, neighbors, self.attention_chunk)
        return h

    @torch.inference_mode()
    def encode_streamed(self, inputs, positions, neighbors, device, chunk=256):
        """Exact layer-wise attention, with full-video states kept on the CPU.

        All original neighbors are used at every layer. Chunking the query rows
        never truncates the receptive field at temporal or spatial boundaries.
        This inference path intentionally uses FP32, matching E013 calibration.
        """
        if self.training or chunk < 1:
            raise ValueError('Streaming requires eval mode and a positive chunk')
        device = torch.device(device)
        x = torch.as_tensor(inputs, dtype=torch.float32, device='cpu')
        p = torch.as_tensor(positions, dtype=torch.float32, device='cpu')
        ix = torch.as_tensor(neighbors, dtype=torch.long, device='cpu')
        n = len(x)
        if not n:
            raise ValueError('Empty graph')
        h = torch.empty((n, self.config['width']), dtype=torch.float32)
        for start in range(0, n, chunk):
            h[start:start+chunk] = self.input(x[start:start+chunk].to(device)).cpu()
        for layer in self.layers:
            qkv = torch.empty((n, 3*self.config['width']), dtype=torch.float32)
            for start in range(0, n, chunk):
                qkv[start:start+chunk] = layer.qkv(layer.norm(h[start:start+chunk].to(device))).cpu()
            result = torch.empty_like(h)
            for start in range(0, n, chunk):
                stop = min(start+chunk, n); ids = ix[start:stop]; safe = ids.clamp_min(0)
                q = qkv[start:stop, :self.config['width']].to(device)
                kv = qkv[safe, self.config['width']:].to(device)
                k, v = kv.chunk(2, dim=-1)
                delta = (p[safe]-p[start:stop, None]).to(device)
                weights = (q[:, None]*k).sum(-1)/math.sqrt(h.shape[-1])
                weights = weights + layer.geometry(delta).squeeze(-1)
                weights = weights.masked_fill(ids.to(device) < 0, -1e4).softmax(-1)
                update = h[start:stop].to(device) + layer.output((weights[..., None]*v).sum(1))
                result[start:stop] = (update + layer.feed(update)).cpu()
            h = result
        return h

    def links(self, h, p, edges):
        a, b = edges.T; delta = p[b, 1:]-p[a, 1:]
        geometry = torch.cat((delta, delta.norm(dim=-1, keepdim=True)), -1)
        return self.link_head(torch.cat((h[a], h[b], (h[a]-h[b]).abs(), h[a]*h[b], geometry), -1)).squeeze(-1)

    def divisions(self, h, p, triples):
        m, a, b = triples.T; da = p[a, 1:]-p[m, 1:]; db = p[b, 1:]-p[m, 1:]
        lengths = torch.stack((da.norm(dim=-1), db.norm(dim=-1)), -1).sort(-1).values
        geometry = torch.cat(((da+db)/2, (da-db).abs(), lengths,
                              (da-db).norm(dim=-1, keepdim=True), (da*db).sum(-1, keepdim=True)), -1)
        return self.division_head(torch.cat((h[m], (h[a]+h[b])/2, (h[a]-h[b]).abs(), h[a]*h[b], geometry), -1)).squeeze(-1)

    def quality(self, h):
        return self.quality_head(h).squeeze(-1)
