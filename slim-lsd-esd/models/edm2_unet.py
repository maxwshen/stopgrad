import math
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- magnitude-preserving helpers ----------

def _multi_axis_l2(x: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
    # L2 norm across dims, with keepdim
    return (x.to(torch.float32).pow(2).sum(dim=dims, keepdim=True)).sqrt()

def mp_normalize(x: torch.Tensor, dims: Optional[Tuple[int, ...]] = None, eps: float = 1e-4) -> torch.Tensor:
    """
    Normalize tensor to unit magnitude along given dims (default: all but dim0).
    Matches the paper/JAX logic: divide by (eps + ||x|| * sqrt(norm_size / x_size)).
    """
    if dims is None:
        dims = tuple(range(1, x.ndim))
    norm = _multi_axis_l2(x, dims).to(torch.float32)
    norm_size = 1
    for d in norm.shape:
        norm_size *= d
    x_size = 1
    for d in x.shape:
        x_size *= d
    scale = torch.sqrt(torch.tensor(norm_size / x_size, dtype=torch.float32, device=x.device))
    denom = eps + norm * scale
    return x / denom.to(x.dtype)

def mp_silu(x: torch.Tensor) -> torch.Tensor:
    # SiLU / 0.596 to preserve unit variance
    return F.silu(x) / 0.596

def mp_sum(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    # magnitude-preserving weighted sum
    y = a * (1.0 - t) + b * t
    return y / math.sqrt((1.0 - t) ** 2 + t ** 2)

def mp_cat(a: torch.Tensor, b: torch.Tensor, dim: int = 1, t: float = 0.5) -> torch.Tensor:
    # magnitude-preserving concatenation that balances contributions by channel counts
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = math.sqrt((Na + Nb) / ((1.0 - t) ** 2 + t ** 2))
    wa = (C / math.sqrt(max(Na, 1.0))) * (1.0 - t)
    wb = (C / math.sqrt(max(Nb, 1.0))) * t
    return torch.cat([a * wa, b * wb], dim=dim)

def _make_depthwise_filter_2d(f: List[float], C: int, dtype, device):
    """
    Expand a 1D filter list into a normalized 2D kernel, depthwise per-channel.
    Returns weight shaped [C, 1, k, k] for grouped conv with groups=C.
    """
    f = torch.tensor(f, dtype=torch.float32, device=device)
    assert f.ndim == 1 and (len(f) % 2 == 0), "filter must have even length"
    f = f / f.sum()
    k2d = torch.outer(f, f)                      # [k, k]
    k2d = k2d[None, None, :, :].to(dtype=dtype)  # [1,1,k,k]
    k2d = k2d.repeat(C, 1, 1, 1)                 # [C,1,k,k]
    return k2d

def resample(x: torch.Tensor, f: List[float] = [1, 1], mode: str = "keep") -> torch.Tensor:
    if mode == "keep":
        return x
    C = x.shape[1]
    weight = _make_depthwise_filter_2d(f, C, dtype=x.dtype, device=x.device)
    L = len(f)
    assert L % 2 == 0, "filter length must be even"
    pad = (L - 1) // 2  # used by down path

    if mode == "down":
        return F.conv2d(x, weight, stride=2, padding=pad, groups=C)

    # --- upsample: EXACT 2x doubling ---
    up_pad = L // 2 - 1         # e.g., L=2 -> 0, L=4 -> 1
    return F.conv_transpose2d(
        x, weight * 4.0, stride=2, padding=up_pad, output_padding=0, groups=C
    )


# ---------- modules ----------

class MPPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10_000.0):
        super().__init__()
        assert (dim // 2) % 2 == 0, "half must be divisible by 2"
        self.dim = dim
        self.max_period = float(max_period)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t = timesteps.reshape(-1).to(torch.float32)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None] * freqs[None, :]
        cos = torch.cos(args) * math.sqrt(2.0)
        sin = torch.sin(args) * math.sqrt(2.0)
        emb = torch.cat([cos, sin], dim=-1).to(timesteps.dtype)
        return emb


class MPConv(nn.Module):
    """
    Magnitude-preserving linear/conv:
    - stores weight as 'mpconv_weight'
    - per-use normalization across fan-in dims
    - scales by (gain / sqrt(w_size)) where w_size = prod(fan-in)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel: Tuple[int, int] = ()):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if kernel:
            kH, kW = kernel
            self.is_linear = False
            shape = (out_channels, in_channels, kH, kW)
        else:
            self.is_linear = True
            shape = (out_channels, in_channels)
        w = torch.randn(*shape) * 1.0
        self.mpconv_weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor, gain: torch.Tensor | float = 1.0) -> torch.Tensor:
        w = self.mpconv_weight.to(torch.float32)
        dims = tuple(range(1, w.ndim))
        w = mp_normalize(w, dims=dims)
        w_size = 1
        for d in w.shape[1:]:
            w_size *= d
        scale = (gain / math.sqrt(max(w_size, 1))).to(x.dtype) if isinstance(gain, torch.Tensor) \
                else (float(gain) / math.sqrt(max(w_size, 1)))
        w = (w.to(x.dtype) * scale)

        if self.is_linear:
            if x.ndim == 2:
                return F.linear(x, w)
            elif x.ndim == 4:
                B, C, H, W = x.shape
                x2 = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
                y2 = F.linear(x2, w)
                y  = y2.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()
                return y
            else:
                raise ValueError(f"MPConv(linear) expected x.ndim in {{2,4}}, got {x.ndim}")
        else:
            k = w.shape[-1]
            pad = k // 2
            return F.conv2d(x, w, stride=1, padding=pad)


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        flavor: str = "enc",
        resample_mode: str = "keep",
        resample_filter: Optional[List[int]] = None,
        attention: bool = False,
        channels_per_head: int = 64,
        dropout: float = 0.0,
        res_balance: float = 0.3,
        attn_balance: float = 0.3,
        clip_act: Optional[int] = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.resample_filter = resample_filter or [1, 1]
        self.has_attention = attention
        self.channels_per_head = channels_per_head
        self.dropout_p = float(dropout)
        self.res_balance = float(res_balance)
        self.attn_balance = float(attn_balance)
        self.clip_act = clip_act

        self.emb_gain = nn.Parameter(torch.zeros(()))

        self.conv_res0 = MPConv(out_channels if flavor == "enc" else in_channels, out_channels, kernel=(3, 3))
        self.emb_linear = MPConv(emb_channels, out_channels)
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=(3, 3))
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else None

        self.conv_skip = MPConv(in_channels, out_channels, kernel=(1, 1)) if in_channels != out_channels else None

        if self.has_attention:
            self.num_heads = max(out_channels // channels_per_head, 1)
            self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=(1, 1))
            self.attn_proj = MPConv(out_channels, out_channels, kernel=(1, 1))
        else:
            self.num_heads = 0
            self.attn_qkv = None
            self.attn_proj = None

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, train: Optional[bool] = None) -> torch.Tensor:
        if train is None:
            train = self.training

        x = resample(x, f=self.resample_filter, mode=self.resample_mode)

        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = mp_normalize(x, dims=(1,))
        y = self.conv_res0(mp_silu(x))
        if emb is None:
            raise ValueError("Block expects an embedding tensor")
        c = self.emb_linear(emb, gain=self.emb_gain) + 1.0
        c = c[:, :, None, None].to(y.dtype)
        y = mp_silu(y * c)

        if train and (self.dropout is not None):
            y = self.dropout(y)
        y = self.conv_res1(y)

        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        if self.num_heads > 0:
            B, C, H, W = x.shape
            y = self.attn_qkv(x)
            y = y.view(B, self.num_heads, -1, 3, H * W)
            y = mp_normalize(y, dims=(2,))
            q = y[:, :, :, 0, :]
            k = y[:, :, :, 1, :]
            v = y[:, :, :, 2, :]
            scale = math.sqrt(q.shape[2])
            attn = torch.einsum('bhci,bhcj->bhij', q, k / scale)
            attn = F.softmax(attn, dim=-1)
            y = torch.einsum('bhij,bhcj->bhci', attn, v)
            y = y.reshape(B, C, H, W)
            y = self.attn_proj(y)
            x = mp_sum(x, y, t=self.attn_balance)

        if self.clip_act is not None:
            x = x.clamp_(-self.clip_act, self.clip_act)
        return x


# ---------- EDM2 U-Net ----------

class EDM2UNet(nn.Module):
    """
    EDM2-style magnitude-preserving U-Net.
    Adapted from the EDM2 paper architecture to match the forward interface
    used by the other nets in this repo: forward(t, u, x, aug_cond=None).
    """
    def __init__(
        self,
        img_resolution: int = 32,
        img_channels: int = 3,
        label_dim: int = 0,
        model_channels: int = 192,
        channel_mult: List[int] = (1, 2, 3, 4),
        channel_mult_noise: Optional[int] = None,
        channel_mult_emb: Optional[int] = None,
        num_blocks: int = 3,
        attn_resolutions: List[int] = (16, 8),
        label_balance: float = 0.5,
        concat_balance: float = 0.5,
        dropout: float = 0.0,
        augment_dim: int = 0,
        block_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.img_resolution = int(img_resolution)
        self.img_channels = int(img_channels)
        self.label_dim = int(label_dim)
        self.model_channels = int(model_channels)
        self.channel_mult = list(channel_mult)
        self.channel_mult_noise = channel_mult_noise
        self.channel_mult_emb = channel_mult_emb
        self.num_blocks = int(num_blocks)
        self.attn_resolutions = set(attn_resolutions)
        self.label_balance = float(label_balance)
        self.concat_balance = float(concat_balance)

        self.block_kwargs = block_kwargs or {}
        self.block_kwargs.setdefault('dropout', dropout)

        # channels per level
        cblock = [self.model_channels * m for m in self.channel_mult]
        cst = (self.model_channels * self.channel_mult_noise) if (self.channel_mult_noise is not None) else cblock[0]
        cemb = (self.model_channels * self.channel_mult_emb) if (self.channel_mult_emb is not None) else max(cblock)
        self.cblock, self.cst, self.cemb = cblock, cst, cemb

        # learned output gain, zero init
        self.out_gain = nn.Parameter(torch.zeros(()))

        self.emb_t_fourier = MPPositionalEmbedding(cst)
        self.emb_u_fourier = MPPositionalEmbedding(cst)
        self.emb_t_linear = MPConv(cst, cemb)
        self.emb_u_linear = MPConv(cst, cemb)

        # optional class label embedding
        if (self.label_dim is not None) and (self.label_dim > 0):
            self.emb_label = MPConv(self.label_dim, cemb)
        else:
            self.emb_label = None

        # optional augmentation embedding
        if augment_dim > 0:
            self.map_augment = MPConv(augment_dim, cemb)
        else:
            self.map_augment = None

        # Encoder
        self.enc = nn.ModuleDict()
        cout = self.img_channels + 1  # +1 for constant channel
        for level, channels in enumerate(cblock):
            res = self.img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                m = MPConv(cin, cout, kernel=(3, 3))
                m.out_channels = cout
                self.enc[f"{res}x{res}_conv"] = m
            else:
                m = Block(cout, cout, cemb, flavor="enc", resample_mode="down",
                          attention=False, **self.block_kwargs)
                m.out_channels = cout
                self.enc[f"{res}x{res}_down"] = m
            for idx in range(self.num_blocks):
                cin = cout
                cout = channels
                m = Block(cin, cout, cemb, flavor="enc",
                          attention=(res in self.attn_resolutions),
                          **self.block_kwargs)
                m.out_channels = cout
                self.enc[f"{res}x{res}_block{idx}"] = m

        # Decoder
        self.dec = nn.ModuleDict()
        skips_out = [blk.out_channels for blk in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = self.img_resolution >> level
            if level == len(cblock) - 1:
                m = Block(cout, cout, cemb, flavor="dec", attention=True, **self.block_kwargs)
                m.out_channels = cout
                self.dec[f"{res}x{res}_in0"] = m
                m = Block(cout, cout, cemb, flavor="dec", **self.block_kwargs)
                m.out_channels = cout
                self.dec[f"{res}x{res}_in1"] = m
            else:
                m = Block(cout, cout, cemb, flavor="dec", resample_mode="up", **self.block_kwargs)
                m.out_channels = cout
                self.dec[f"{res}x{res}_up"] = m
            for idx in range(self.num_blocks + 1):
                skip_c = skips_out.pop()
                cin = cout + skip_c
                cout = channels
                m = Block(cin, cout, cemb, flavor="dec",
                          attention=(res in self.attn_resolutions),
                          **self.block_kwargs)
                m.out_channels = cout
                self.dec[f"{res}x{res}_block{idx}"] = m

        self.out_conv = MPConv(cout, self.img_channels, kernel=(3, 3))

    def _prep_embed(self, standalone_time, diff_time, aug_cond):
        emb_t = self.emb_t_linear(self.emb_t_fourier(standalone_time))
        emb_u = self.emb_u_linear(self.emb_u_fourier(diff_time))
        emb = mp_sum(emb_t, emb_u, t=0.5)

        if (self.emb_label is not None):
            raise NotImplementedError("Class label conditioning not yet wired up in this repo.")

        if self.map_augment is not None and aug_cond is not None:
            aug_emb = self.map_augment(aug_cond)
            emb = emb + aug_emb

        return mp_silu(emb)

    def forward(self, t, u, x, aug_cond=None):

        B, C, H, W = x.shape


        emb = self._prep_embed(t, u-t, aug_cond)

        # Encoder
        x_in = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        h = x_in
        for name, block in self.enc.items():
            if "conv" in name:
                h = block(h)
            else:
                h = block(h, emb)
            skips.append(h)

        # Decoder
        for name, block in self.dec.items():
            if "block" in name:
                skip = skips.pop()
                assert skip.shape[2:] == h.shape[2:], f"skip {skip.shape} vs h {h.shape} at {name}"
                h = mp_cat(h, skip, dim=1, t=self.concat_balance)
            h = block(h, emb)

        # Final conv with learned gain
        out = self.out_conv(h, gain=self.out_gain)
        return out
