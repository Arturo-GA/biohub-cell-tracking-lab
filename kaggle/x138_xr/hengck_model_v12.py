# Model definitions from hengck23/model_v12.py, public Kaggle End2EndCellLinker.
# https://www.kaggle.com/code/hengck23/end2end-cell-linker-raw-edge-ja-0-9-no-ilp
# Only imports and definitions retained; demonstration code removed.
from typing import Any

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as grad_ckpt

class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def sample_one_feature_at_zyx(
    feat, zyx, volume_shape, mask=None
):
    Z, Y, X = volume_shape
    gz = 2.0 * (zyx[..., 0]+0.5) / Z - 1
    gy = 2.0 * (zyx[..., 1]+0.5) / Y - 1
    gx = 2.0 * (zyx[..., 2]+0.5) / X - 1
    grid = torch.stack([gx, gy, gz], dim=-1)[:, :, None, None, :]

    sampled = F.grid_sample(
        feat,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    sampled =  sampled[:, :, :, 0, 0].transpose( 1, 2 )
    if mask is not None:
        m = mask[..., None].to(feat.dtype)
        sampled = m*sampled
    return sampled

def sample_pyr_feature_at_zyx(
    pyramid, zyx, volume_shape, mask=None,
):
    Z, Y, X = volume_shape
    gz = 2.0 * (zyx[..., 0]+0.5) / Z - 1
    gy = 2.0 * (zyx[..., 1]+0.5) / Y - 1
    gx = 2.0 * (zyx[..., 2]+0.5) / X - 1
    grid = torch.stack([gx, gy, gz], dim=-1)[:, :, None, None, :]

    if mask is not None:
        m = mask[..., None].to(pyramid[0].dtype)

    sampled =[]
    for feat in pyramid:
        s = F.grid_sample(
            feat,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        s = s[:, :, :, 0, 0].transpose( 1, 2 )
        if mask is not None:
            s = s * m
        sampled.append(s)
    return sampled

def max_pool_peak(
    node_logit, CFG
):
    node_prob = torch.sigmoid(node_logit.float())
    pooled = torch.nn.functional.max_pool3d(
            node_prob,
            kernel_size=CFG.node_peak_kernel,
            stride=1,
            padding=CFG.node_peak_kernel // 2,
        )
    keep = (node_prob >= pooled) & ( node_prob >= CFG.node_peak_threshold )

    batch_size = len(node_prob)
    return [
        keep[ b, 0 ].nonzero( as_tuple=False).float()
        for b in range(batch_size)
    ]

def extract_node_peak(
    node_logit,  CFG
):
    peak = max_pool_peak(node_logit, CFG) #zyx
    max_allowed = CFG.max_detected_nodes

    for b, pk in enumerate(peak):
        if len(pk) == 0:
            flat_index = node_logit[b, 0].float().argmax()
            shape = node_logit.shape[-3:]
            z = flat_index // (shape[1] * shape[2])
            remainder = flat_index % (shape[1] * shape[2])
            y = remainder // shape[2]
            x = remainder % shape[2]
            peak[b] = torch.stack([z, y, x]).float()[None]
            pk = peak[b]

        if max_allowed is not None and len(pk) > max_allowed:
            index = pk.long()
            score = node_logit[b, 0, index[:, 0], index[:, 1], index[:, 2]].float()
            peak[b] = pk[score.topk(max_allowed).indices] #todo: issue of using node_logit and refine_logit

    batch_size = len(peak)
    N_max = max(len(pk) for pk in peak)
    coord = node_logit.new_zeros((batch_size, N_max, 3), dtype=torch.float32)
    logit = torch.zeros((batch_size, N_max), dtype=torch.float32, device=node_logit.device)
    mask  = torch.zeros((batch_size, N_max), dtype=torch.bool, device=node_logit.device)
    count = []
    for b, pk in enumerate(peak):
        N = len(pk)
        count.append(N)

        index = pk.long()
        coord[b, :N] = pk.to(node_logit.device)
        logit[b, :N] = node_logit[b,0,index[:,0],index[:,1],index[:,2]]
        mask[b, :N] = True

    return coord, logit, mask, count

def refine_node_peak(
    coord, logit, refinement,
):
    # refinement: (B, 4, Z, Y, X)
    # coord:      (B, N, 3)
    # logit:      (B, N)

    B, _, Z, Y, X = refinement.shape

    index = coord.round().long()
    iz = index[..., 0].clamp(0, Z - 1)
    iy = index[..., 1].clamp(0, Y - 1)
    ix = index[..., 2].clamp(0, X - 1)
    batch_index = torch.arange( B, device=refinement.device,)[:, None]

    # Convert to (B, Z, Y, X, 4), then sample each node.
    r = refinement.permute(0, 2, 3, 4, 1)
    r = r[batch_index, iz, iy, ix]  # (B, N, 4)

    dzyx   = r[..., :3]
    dlogit = r[..., 3]
    refine_coord = coord.detach() + dzyx
    refine_logit = logit.detach() + dlogit
    #todo: handle out of boundary after update?

    return refine_coord, refine_logit

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d( out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)

        self.drop = (
            nn.Dropout3d(dropout)
            if dropout > 0
            else nn.Identity()
        )

        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(
                nn.Conv3d(
                    in_ch,
                    out_ch,
                    1,
                    bias=False,
                ),
                nn.BatchNorm3d(out_ch),
            )
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)

        x = self.act( self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        return self.act(x + identity)

class UNet3D(nn.Module):
    def __init__(
        self, CFG=None,

        # channel=(64, 128, 256),
        # gradient_checkpointing=True,
        # dropout=0.0,
    ):
        super().__init__()
        self.CFG=CFG
        c0, c1, c2 = CFG.channel
        dropout = CFG.dropout
        self.gradient_checkpointing = CFG.gradient_checkpointing

        self.enc0  = ResBlock3D(1, c0, dropout)
        self.pool0 = nn.MaxPool3d(2, 2)
        self.enc1  = ResBlock3D(c0, c1, dropout)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.enc2 = ResBlock3D(c1, c2, dropout)

        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2, bias=False)
        self.dec1 = ResBlock3D(c1 + c1, c1, dropout)
        self.up0 = nn.ConvTranspose3d(c1, c0, 2, stride=2, bias=False)
        self.dec0 = ResBlock3D(c0 + c0, c0, dropout)

        head_mid = max(c0 // 2, 32)

        self.node_head = nn.Sequential(
            ResBlock3D(c0, c0),
            nn.Conv3d(c0, head_mid, 3, padding=1, bias=False),
            nn.BatchNorm3d(head_mid),
            nn.ReLU(inplace=True),
            nn.Conv3d(head_mid, 1, 1, ),
        )

        self.refine_head = nn.Sequential(
            ResBlock3D(c0, c0),
            nn.Conv3d(c0, head_mid, 3, padding=1, bias=False),
            nn.BatchNorm3d(head_mid),
            nn.ReLU(inplace=True),
            nn.Conv3d(head_mid, 4, 1, ),  # dzyx dp
        )

        final_conv = self.refine_head[-1]
        nn.init.zeros_(final_conv.weight)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)


    def _run(self, block, x):
        if ( self.gradient_checkpointing
            and self.training
        ):
            return grad_ckpt( block, x, use_reentrant=False, )
        return block(x)

    def make_feature(self, volume):
        x = volume.float() # B,1,Z,X,Y

        e0 = self._run(self.enc0, x)
        e1 = self._run(self.enc1, self.pool0(e0))
        e2 = self._run(self.enc2, self.pool1(e1))

        x = self.up1(e2)
        if x.shape[2:] != e1.shape[2:]:
            x = F.interpolate(
                x, size=e1.shape[2:], mode="trilinear", align_corners=False,
            )
        d1 = self._run( self.dec1, torch.cat([x, e1], dim=1))

        x = self.up0(d1)
        if x.shape[2:] != e0.shape[2:]:
            x = F.interpolate(
                x, size=e0.shape[2:], mode="trilinear", align_corners=False,
            )
        d0 = self._run( self.dec0, torch.cat([x, e0], dim=1) )
        x = d0

        return [e0,e1,e2,d0,d1], x

    def forward(self, volume):
        CFG = self.CFG

        [e0,e1,e2,d0,d1], x = self.make_feature(volume)
        node_logit = self.node_head(x)
        refinement = self.refine_head(x)

        #todo: discuss use encoder or decoder feature? use last feature?
        #    f0,f1,f2 = e0,e1,e2
        # or f0,f1,f2 = e2,d0,d1
        pyr = [d0,d1,e2,]
        #print("pyr",[p.shape for p in pyr]) #(64, 128, 256)

        peak_zyx, peak_logit, mask, count = extract_node_peak(node_logit, CFG)
        refine_zyx, refine_logit = refine_node_peak(peak_zyx, peak_logit, refinement)
        node_feature = sample_pyr_feature_at_zyx(pyr, refine_zyx, volume.shape[-3:], mask)

        return DotDict(
            pyr=pyr,
            node_logit=node_logit,
            refinement=refinement,
            peak_zyx=peak_zyx,
            peak_logit=peak_logit,
            refine_zyx=refine_zyx,
            refine_logit=refine_logit,
            node_feature=node_feature,
            mask=mask,
            count=count,
        )

class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim=256,
        n_heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
        attention_chunk_size=256,
    ):
        super().__init__()

        self.attention_chunk_size = attention_chunk_size
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim,n_heads,dropout=dropout, batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def _run_chunk(
        self, q, kvn, key_padding_mask,
    ):
        qn = self.norm_q(q)
        attn_out, _ = self.attn(
            qn, kvn, kvn,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = q + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x

    def forward(
        self, q, kv, kv_mask=None,
    ):
        key_padding_mask = ~kv_mask.bool() if kv_mask is not None else None

        kvn = self.norm_kv(kv)
        Nq = q.shape[1]
        chunk = self.attention_chunk_size

        if (
            chunk is None
            or chunk <= 0
            or Nq <= chunk
        ):
            return self._run_chunk( q, kvn, key_padding_mask)

        out = []
        for i in range( 0, Nq, chunk ):
            out.append(
                self._run_chunk(  q[:, i:i + chunk], kvn,  key_padding_mask,)
            )
        return torch.cat(out, dim=1)

class SelfCrossBlock(nn.Module):
    def __init__(
        self,
        dim=256,
        n_heads=4,
        dropout=0.1,
        attention_chunk_size=256,
    ):
        super().__init__()

        self.self_attn = AttentionBlock(
            dim=dim,
            n_heads=n_heads,
            dropout=dropout,
            attention_chunk_size=attention_chunk_size,
        )
        self.cross_attn = AttentionBlock(
            dim=dim,
            n_heads=n_heads,
            dropout=dropout,
            attention_chunk_size=attention_chunk_size,
        )

    def forward(
        self, h0, h1, mask0=None, mask1=None,
    ):
        h0 = self.self_attn( h0, h0, mask0 )
        h1 = self.self_attn( h1, h1, mask1 )

        old0, old1 = h0, h1
        h0 = self.cross_attn(old0, old1, mask1)
        h1 = self.cross_attn(old1, old0, mask0)
        return h0, h1

class NodeEncoder(nn.Module):
    def __init__(
        self,
        feat_dims=(64, 128, 256),
        hidden_dim=256,
    ):
        super().__init__()

        self.visual_proj = nn.Sequential(
            nn.Linear(sum(feat_dims),  hidden_dim ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.position_mlp = nn.Sequential(
            nn.Linear(9, 64),
            nn.GELU(),
            nn.Linear( 64, hidden_dim,),
        )

    def forward(
        self, features, zyx, volume_shape,
    ):
        visual = self.visual_proj( torch.cat( features,  dim=-1,) )

        shape = torch.as_tensor( volume_shape, dtype=zyx.dtype, device=zyx.device,).clamp_min(2)
        p = zyx / (shape - 1.0)
        z = p[..., 0:1]
        y = p[..., 1:2]
        x = p[..., 2:3]
        boundary = torch.cat( [ z, 1 - z, y, 1 - y, x, 1 - x, ], dim=-1)
        pos = self.position_mlp( torch.cat(  [p, boundary], dim=-1, ))
        return visual + pos

class MetricHead(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        embed_dim=128,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear( hidden_dim, embed_dim),
        )

    def forward(self, x):
        return F.normalize( self.proj(x), dim=-1 )

class PairHead(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        pair_hidden=256,
        pair_chunk_size=32,
        dropout=0.1,
    ):
        super().__init__()

        self.pair_chunk_size =  pair_chunk_size
        pair_dim = ( hidden_dim * 4 + 3 + 1 + 1 )
        self.mlp = nn.Sequential(
            nn.Linear( pair_dim, pair_hidden ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear( pair_hidden, pair_hidden // 2 ),
            nn.GELU(),
            nn.Linear( pair_hidden // 2,  1 ),
        )

    def forward(
        self, h0, h1, e0, e1, zyx0, zyx1,
        spatial_scale=100.0, #todo: too large????
    ):
        B, N0, D = h0.shape
        N1 = h1.shape[1]
        chunk =  self.pair_chunk_size or N0

        cosine = torch.matmul(  e0, e1.transpose(-1, -2) ) #cosine similarity

        out = []
        for i in range(0, N0, chunk):
            j = min( i + chunk, N0)

            a0 = h0[:, i:j]
            c0 = zyx0[:, i:j]

            nc = j - i
            a = a0[:, :, None, :].expand( -1, -1, N1, -1 )
            b = h1[:, None, :, :].expand( -1, nc, -1, -1 )

            dp = ( zyx1[:, None, :, :] - c0[:, :, None, :] ) / spatial_scale
            dist = torch.linalg.vector_norm(dp, dim=-1, keepdim=True )
            cos = cosine[ :, i:j, :, None ]

            pair = torch.cat(
                [
                    a,
                    b,
                    a - b,
                    a * b, #a,b are appearance
                    dp,   #dir
                    dist, #dist
                    cos,  #similarity
                ],
                dim=-1,
            )
            out.append( self.mlp(pair).squeeze(-1) )
        return torch.cat( out, dim=1 )

class CellEdgeTransformer(nn.Module):
    def __init__(
        self, CFG,
        # feat_dims=(64, 128, 256),
        # hidden_dim=256,
        # embed_dim=128,
        # n_heads=4,
        # n_layers=4,
        # dropout=0.1,
    ):
        super().__init__()
        pair_chunk_size=32
        attention_chunk_size=256
        feat_dims=CFG.feat_dims #(64, 128, 256),
        hidden_dim=CFG.hidden_dim #256,
        embed_dim=CFG.embed_dim #128,
        n_heads=CFG.n_heads #4,
        n_layers=CFG.n_layers #4,
        dropout=CFG.dropout #0.1,


        self.node_encoder = NodeEncoder(
            feat_dims, hidden_dim,
        )

        self.blocks = nn.ModuleList([
            SelfCrossBlock(
                dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
                attention_chunk_size=attention_chunk_size,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim )
        self.metric_head = MetricHead(hidden_dim, embed_dim)

        self.pair_head = PairHead(
            hidden_dim=hidden_dim,
            pair_hidden=hidden_dim,
            pair_chunk_size=pair_chunk_size,
            dropout=dropout,
        )

    def forward(
        self, feat0, feat1, zyx0, zyx1, mask0, mask1,
        volume_shape,
    ):
        h0 = self.node_encoder( feat0,  zyx0, volume_shape )
        h1 = self.node_encoder( feat1, zyx1,  volume_shape )

        for block in self.blocks:
            h0, h1 = block( h0, h1, mask0, mask1 )

        h0 = self.final_norm(h0)
        h1 = self.final_norm(h1)

        h0 = h0 * mask0[..., None  ].to(h0.dtype)
        h1 = h1 * mask1[..., None  ].to(h1.dtype)
        e0 = self.metric_head(h0)
        e1 = self.metric_head(h1)

        metric_score = torch.matmul( e0, e1.transpose(-1, -2),)
        edge_logit = self.pair_head( h0, h1, e0, e1, zyx0, zyx1 )

        return DotDict(
            h0 = h0,
            h1 = h1,
            embedding0 = e0,
            embedding1 = e1,
            metric_score = metric_score,
            edge_logit = edge_logit,
        )

class End2EndCellLinker(nn.Module):
    def __init__(self, CFG=None):
        super().__init__()
        self.unet = UNet3D(CFG.unet_cfg)
        self.linker = CellEdgeTransformer(CFG.tx_cfg)

    def forward(
        self,
        volume0,
        volume1,
    ): #forward coded for completely only. we usually use self.unet(), self.linker() in custome inference code
        B,_1_,Z,Y,X = volume0.shape
        out0 = self.unet(volume0)
        out1 = self.unet(volume1)
        out01 = self.linker(
            out0.node_feature, out1.node_feature,
            out0.refine_zyx, out1.refine_zyx,
            out0.mask, out1.mask, (Z, Y, X),
        )
        return out0, out1, out01
