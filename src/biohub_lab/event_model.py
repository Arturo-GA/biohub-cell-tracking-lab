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
        self.input = nn.Sequential(nn.LayerNorm(72), nn.Linear(72, width), nn.SiLU())
        self.layers = nn.ModuleList([LocalAttention(width) for _ in range(layers)])
        self.link_head = mlp(4*width+4, width)
        self.division_head = mlp(4*width+10, width)
        self.quality_head = mlp(width, width)

    def encode(self, inputs, positions, neighbors):
        h = self.input(inputs)
        for layer in self.layers:
            h = layer(h, positions, neighbors)
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
