
from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

def default_edm2_init(module):
    if hasattr(module, 'weight') and module.weight is not None:
        if module.weight.ndim >= 2:
            th.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
    if hasattr(module, 'bias') and module.bias is not None:
        th.nn.init.zeros_(module.bias)

def scale_shift_init(module):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, 0)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, 0)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        net = nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        net = nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        net = nn.Conv3d(*args, **kwargs)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")
    net.apply(default_edm2_init)
    return net

def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    m = nn.Linear(*args, **kwargs)
    m.apply(default_edm2_init)
    # saining init
    #nn.init.xavier_uniform_(m.weight)
    #if m.bias is not None:
    #    nn.init.constant_(m.bias, 0)
    return m

def scale_shift_linear(*args, **kwargs):
    """
    Create a linear module with saining init
    """
    m = nn.Linear(*args, **kwargs)
    m.apply(scale_shift_init)
    return m

def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)

class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

class PixelNorm(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.epsilon = eps

    def forward(self, x):
        # x: (batch, channels, height, width)
        norm_squared = th.mean(x**2, dim=1, keepdim=True)  # (batch, 1, height, width)
        eps = th.tensor(self.epsilon, dtype=x.dtype, device=x.device)
        denom = th.sqrt(norm_squared + eps)
        return x / denom

def append_dims(t, dims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))

class RMSNorm(nn.Module):
    def __init__(self, dim, scale = True, normalize_dim = 1, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(th.ones(dim)) if scale else 1
        self.scale = scale
        self.normalize_dim = normalize_dim
        self.eps = eps

    def forward(self, x):
        normalize_dim = self.normalize_dim
        scale = append_dims(self.g, x.ndim - self.normalize_dim - 1) if self.scale else 1
        return F.normalize(x, dim = normalize_dim, eps=self.eps) * scale * (x.shape[normalize_dim] ** 0.5)


class ChannelLayerNorm(nn.Module):
    """
    LayerNorm over the channel vector at every position, for any tensor rank ≥ 3.
    The channel axis is defined by `normalize_dim` (default 1 for (N,C,⋯)).
    """
    def __init__(self, n_channels: int,
                 eps: float = 1e-5,
                 affine: bool = True,
                 normalize_dim: int = 1):
        super().__init__()
        self.normalize_dim = normalize_dim
        self.ln = nn.LayerNorm(n_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        # Permute so that the channel axis becomes the *last* axis
        perm = [d for d in range(x.ndim) if d != self.normalize_dim] + [self.normalize_dim]
        x_perm = x.permute(perm)

        # Apply LayerNorm on the last axis (the channels)
        x_norm = self.ln(x_perm)

        # Invert the permutation to restore original layout
        inv_perm = list(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        return x_norm.permute(inv_perm).contiguous()





class CustomGroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, affine=False, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(th.ones(num_channels))
            self.bias = nn.Parameter(th.zeros(num_channels))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):

        
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None

        x = th.nn.functional.group_norm(x, num_groups=self.num_groups, weight=w, bias=b, eps=self.eps)
        # this is the meanflow one
        #x = group_norm(
        #    x, 
        #    num_groups=self.num_groups, 
        #    weight=self.weight.to(x.dtype) if self.affine else None, 
        #    bias=self.bias.to(x.dtype) if self.affine else None, 
        #    eps=self.eps
        #)
        return x


def get_norm(ch, norm_name, scale=True):
   
    if norm_name == 'rms_norm':
        return RMSNorm(dim=ch, scale=scale, eps=1e-4)
    elif norm_name == 'group_norm':
        return CustomGroupNorm(num_groups=32, num_channels=ch, affine=scale, eps=1e-4)
    elif norm_name == 'instance_norm':
        return CustomGroupNorm(num_groups=ch, num_channels=ch, affine=scale, eps=1e-4) 
    elif norm_name == 'layer_norm':
        return ChannelLayerNorm(n_channels=ch, affine=scale, eps=1e-4)
    elif norm_name == 'pixel_norm':
        return PixelNorm(eps=1e-4)
    elif norm_name == 'no_norm':
        return nn.Identity()
    else:
        assert False

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims

        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:

            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        use_pixelnorm=False,
        norm_name='group_norm',
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        assert not use_checkpoint
        self.use_scale_shift_norm = True
        self.use_pixelnorm = use_pixelnorm

        conv1 = conv_nd(dims, channels, self.out_channels, 3, padding=1)

        self.in_layers = nn.Sequential(
            get_norm(channels, norm_name),
            nn.SiLU(),
            conv1,
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            scale_shift_linear(
                emb_channels,
                2 * self.out_channels 
            ),
        )

        conv2 = conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)

        # scale is false for rmsnorm since norm(x) is scaled and shifted
        self.out_layers = nn.Sequential(
            get_norm(self.out_channels, norm_name, scale=False),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv2),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            conv3 = conv_nd(dims, channels, self.out_channels, 3, padding=1)
            self.skip_connection = conv3
        else:
            conv3 = conv_nd(dims, channels, self.out_channels, 1)
            self.skip_connection = conv3





        if use_pixelnorm:
            self.scale_pixnorm = PixelNorm(eps=1e-4) 
            self.shift_pixnorm = PixelNorm(eps=1e-4) 
        else:
            self.scale_pixnorm = nn.Identity()
            self.shift_pixnorm = nn.Identity()

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            scale = self.scale_pixnorm(scale)
            shift = self.shift_pixnorm(shift)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        norm_name='group_norm',
        which = None,
    ):
        super().__init__()
        self.which = which
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        
        self.norm = get_norm(channels, norm_name)

        self.qkv = conv_nd(1, channels, channels * 3, 1)

        # split qkv before split heads
        self.attention = OldQKVAttention(self.num_heads)
        #else:
        #    assert False
        #    # split heads before split qkv
        #    self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    # TODO FOR DIV NEED TO MAKE THIS FALSE OR MAKE IT USE_CHECKPOINT
    # MADE THIS FALSE FOR DDT
    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), False)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops = model.total_ops + th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        
        # TODO does it fix inplace
        #weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        weight = th.softmax(weight, dim=-1)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class OldQKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))

        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
       
        # TODO does it fix implace
        #weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        weight = th.softmax(weight, dim=-1)

        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))

        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

def next_power_of_2(x):
    return 1 if x == 0 else 2**((x - 1).bit_length())


class TorchQKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):

        # qkv: [B, 3 * H * C, T]
        B, width, T_orig = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)

        # Reshape to [B, 3, H, ch, T]
        qkv = qkv.view(B, 3, self.n_heads, ch, T_orig)
        q, k, v = qkv.unbind(dim=1)  # each is [B, H, ch, T]

        # Transpose to [B, H, T, ch]
        q = q.transpose(-2, -1)  # [B, H, T, ch]
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        T = q.shape[2]
        #T_pad = (8 - T % 8) % 8
        T_target = next_power_of_2(T_orig)
        T_pad = T_target - T_orig
        if T_pad > 0:
            pad = (0, 0, 0, T_pad)  # pad sequence dim only
            q = F.pad(q, pad)
            k = F.pad(k, pad)
            v = F.pad(v, pad)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        if T_pad > 0:
            out = out[:, :, :T, :]  # trim back to original T

        # Transpose back to [B, H, ch, T]
        out = out.transpose(-2, -1).contiguous().view(B, -1, T_orig)
        return out



class FourierPositionalEmbedding(nn.Module):
    def __init__(self, dim, log_space_t_freq=True, max_period=10000, scale=0.02):
        """
        :param dim: total embedding dimension (excluding the extra appended t)
        :param log_space_t_freq: whether to use log-spaced frequencies
        :param max_period: used for log spacing
        :param scale: used for linear spacing
        """
        super().__init__()
        self.dim = dim
        self.log_space_t_freq = log_space_t_freq
        self.max_period = max_period
        self.scale = scale
        half_dim = dim // 2
        indices = th.arange(half_dim)

        if log_space_t_freq:
            freqs = th.exp(-math.log(max_period) * indices / half_dim)
        else:
            freqs = 2 * math.pi * scale * indices

        #self.freqs = nn.Parameter(freqs, requires_grad=False)
        #self.register_buffer("freqs", freqs, persistent=False)
        # after you know your module dtype (e.g., in __init__)
        self.register_buffer("freqs", freqs.to(th.get_default_dtype()))


    def forward(self, t):
        """
        :param t: Tensor of shape [B] or [B, 1]
        :return: Tensor of shape [B, dim (+1 if dim is even) + 1] with t appended
        """
        #f = self.freqs.to(t.dtype)
        f = self.freqs.clone().to(t.dtype)
        t_work = t.clone()[:, None]
        tf = t_work*f[None, :]
        return th.cat([tf.cos(), tf.sin(), t_work.clone()], dim=-1)


class OneTimeEmbedder(nn.Module):
    def __init__(self, fourier_dim, hidden_dim, out_dim, log_space_t_freq, fourier_scale):
        super().__init__()
        self.fourier_pos_embedding = FourierPositionalEmbedding(
            dim = fourier_dim,
            log_space_t_freq = log_space_t_freq,
            max_period = 10000,
            scale = fourier_scale,
        )
        net_in_dim = int(fourier_dim + 1)
        self.log_space_t_freq = log_space_t_freq
        self.net = nn.Sequential(
            linear(net_in_dim, hidden_dim),
            nn.SiLU(),
            linear(hidden_dim, out_dim),
        )
    def forward(self, t):    
        emb = self.fourier_pos_embedding(t)
        emb = self.net(emb)
        return emb

class TwoTimeEmbedder(nn.Module):
    def __init__(self, fourier_dim, hidden_dim, out_dim, log_space_t_freq, fourier_scale):
        super().__init__()
        self.fourier_pos_embedding = FourierPositionalEmbedding(
            dim = fourier_dim,
            log_space_t_freq = log_space_t_freq,
            max_period = 10000,
            scale = fourier_scale,
        )
        net_in_dim = int(2*(fourier_dim+1))
        self.log_space_t_freq = log_space_t_freq
        self.net = nn.Sequential(
            linear(net_in_dim, hidden_dim),
            nn.SiLU(),
            #MPSiLU() if forward_pass_weight_norm else nn.SiLU(),
            linear(hidden_dim, out_dim),
        )
    def forward(self, t, u):    
        ft = self.fourier_pos_embedding(t)
        fu = self.fourier_pos_embedding(u)
        args = th.cat([ft, fu], dim=-1)
        return self.net(args)



#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (th.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * th.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (th.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * th.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')




class Linear(th.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = th.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = th.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x



def attn_bools_to_dhariwal_attn_res(bools, base_res=32, num_downs=None):
    """
    Convert a list of booleans (True=apply attention) into the list of
    downsampling factors used in the Dhariwal et al. (Diffusion) architecture.

    Each stage downsamples by 2× in spatial resolution.

    Example:
        bools = [True, False, True, False, True]
        -> (1, 4, 16) for 32×32 inputs.
    """
    if num_downs is None:
        num_downs = len(bools)

    # compute resolutions per block
    res = [base_res // (2 ** i) for i in range(num_downs)]
    factors = [base_res // r for r in res if r > 0]                                                                                                                                                                                                                                                                                                                                         

    attn_res = tuple(f for f, b in zip(factors, bools) if b)
    return attn_res


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """
    def __init__(self, channel_mult, attention_resolutions, dropout=0.2, augment_dim=0):
        super().__init__()
        image_size=32
        model_channels=256
        in_channels=3
        out_channels=3
        num_res_blocks=2
        num_heads=8
        num_head_channels=64
        num_heads_upsample=-1
        dims=2
        use_checkpoint=False
        log_space_t_freq=True
        use_pixelnorm=False
        fourier_scale=0.02
        norm_name='group_norm'
        keep_dim=False
        num_times=2
        init_zero=dict(init_weight=0)
        self.updown = True
        self.num_times = num_times
        self.resblock_updown = True

        if keep_dim:
            self.resblock_updown = True
            self.updown = False

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = True #conv_resample
        #self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        assert not use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_pixelnorm = use_pixelnorm

        time_embed_dim = model_channels * 4


        if augment_dim:
            self.map_augment = Linear(
                in_features=augment_dim, 
                out_features=time_embed_dim, #model_channels, 
                bias=False, 
                **init_zero
            ) 
        else:
            self.map_augment = None


        if self.num_times == 1:


            self.time_embed = OneTimeEmbedder(
                fourier_dim = model_channels,
                hidden_dim = time_embed_dim,
                out_dim = time_embed_dim,
                log_space_t_freq = log_space_t_freq,
                fourier_scale = fourier_scale,
            )


        elif self.num_times == 2:

            self.time_embed = TwoTimeEmbedder(
                fourier_dim = model_channels,
                hidden_dim = time_embed_dim,
                out_dim = time_embed_dim,
                log_space_t_freq = log_space_t_freq,
                fourier_scale = fourier_scale,
            )


        else:
            assert False


        #if (self.num_classes is not None) and self.num_classes > 0:
        #    self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        #else:
        
        self.label_emb = None
        ch = input_ch = int(channel_mult[0] * model_channels)

        conv = conv_nd(dims, in_channels, ch, 3, padding=1)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv)]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_pixelnorm = use_pixelnorm,
                        norm_name = norm_name,
                    )
                ]
                ch = int(mult * model_channels)

                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            norm_name = norm_name,
                            which = 'down',
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size = self._feature_size + ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            down=self.updown,
                            use_pixelnorm = use_pixelnorm,
                            norm_name = norm_name,
                        )
                        if self.resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds = ds * 2
                self._feature_size = self._feature_size + ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_pixelnorm = use_pixelnorm,
                norm_name = norm_name,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                norm_name = norm_name,
                which = 'middle',
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_pixelnorm = use_pixelnorm,
                norm_name = norm_name,
            ),
        )
        self._feature_size = self._feature_size + ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_pixelnorm = use_pixelnorm,
                        norm_name = norm_name,
                    )
                ]
                ch = int(model_channels * mult)

                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            norm_name = norm_name,
                            which = 'up',
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            up=self.updown,
                            use_pixelnorm = use_pixelnorm,
                            norm_name = norm_name,
                        )
                        if self.resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size = self._feature_size + ch
        
        out_conv = conv_nd(dims, input_ch, out_channels, 3, padding=1)

        self.out = nn.Sequential(
            get_norm(ch, norm_name),
            nn.SiLU(),
            zero_module(out_conv),
        )
    
    def forward(self, t, u, x, aug_cond=None):

        augment_labels = aug_cond

        
        diff_time = u - t


        hs = []

        if self.num_times == 1:
            time_emb = self.time_embed(t)

        elif self.num_times == 2:
            time_emb = self.time_embed(t, u-t)
        
        else:
            assert False

        #if self.label_emb is not None:
        #    assert y.shape == (x.shape[0],)
        #    emb_both = time_emb + self.label_emb(y)
        #else:
        
        emb_both = time_emb

        # added for comparison to meanflow
        if self.map_augment is not None and augment_labels is not None:
            emb_both = emb_both + self.map_augment(augment_labels)
        
        h = x
        for module in self.input_blocks:
            h = module(h, emb_both)
            hs.append(h)
        h = self.middle_block(h, emb_both)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb_both)
        out = self.out(h)
        assert out.shape == x.shape

        return out

