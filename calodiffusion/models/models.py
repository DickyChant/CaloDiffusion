# some pytorch modules & useful functions
from einops import rearrange, repeat
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from inspect import isfunction
from functools import partial


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def pad_arrays(arrays, pad_value=0.0):
    """Pad jagged arrays to the maximum shape along the second dimension."""
    # Determine the maximum size along the jagged dimension
    max_len = max(array.shape[1] for array in arrays)

    # Pad each array
    padded_arrays = []
    for array in arrays:
        pad_size = max_len - array.shape[1]
        padded_array = np.pad(array, ((0, 0), (0, pad_size), (0, 0)), constant_values=pad_value)
        padded_arrays.append(padded_array)
    return np.array(padded_arrays)


def cosine_beta_schedule(nsteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    x = torch.linspace(0, nsteps, nsteps+1)
    alphas_cumprod = torch.cos(((x / nsteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class CylindricalConvTrans(nn.Module):
    # assumes format of channels,zbin,phi_bin,rbin
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size=(3, 4, 4),
        stride=(1, 2, 2),
        groups=1,
        padding=1,
        output_padding=0,
    ):
        super().__init__()
        if type(padding) != int:
            self.padding_orig = copy.copy(padding)
            padding = list(padding)
        else:
            padding = [padding] * 3
            self.padding_orig = copy.copy(padding)

        padding[1] = kernel_size[1] - 1
        self.convTrans = nn.ConvTranspose3d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

    def forward(self, x):
        # out size is : O = (i-1)*S + K - 2P
        # to achieve 'same' use padding P = ((S-1)*W-S+F)/2, with F = filter size, S = stride, W = input size
        # pad last dim with nothing, 2nd to last dim is circular one
        circ_pad = self.padding_orig[1]
        x = F.pad(x, pad=(0, 0, circ_pad, circ_pad, 0, 0), mode="circular")
        x = self.convTrans(x)
        return x


class CylindricalConv(nn.Module):
    # assumes format of channels,zbin,phi_bin,rbin
    def __init__(
        self, dim_in, dim_out, kernel_size=3, stride=1, groups=1, padding=0, bias=True
    ):
        super().__init__()
        if type(padding) != int:
            self.padding_orig = copy.copy(padding)
            padding = list(padding)
            padding[1] = 0
        else:
            padding = [padding] * 3
            self.padding_orig = copy.copy(padding)
            padding[1] = 0
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        # to achieve 'same' use padding P = ((S-1)*W-S+F)/2, with F = filter size, S = stride, W = input size
        # pad last dim with nothing, 2nd to last dim is circular one
        circ_pad = self.padding_orig[1]
        x = F.pad(x, pad=(0, 0, circ_pad, circ_pad, 0, 0), mode="circular")
        x = self.conv(x)
        return x


def zero_module(module):
    # make model parameters zero
    for p in module.parameters():
        p.detach().zero_()
    return module


def make_zero_conv(dim, cylindrical=True):
    # Make a 'zero convolution' layer
    return zero_module(CylindricalConv(dim, dim, kernel_size=1, padding=0))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class ScalarAddLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(1e-6))

    def forward(self, x1, x2):
        # print("Mu", self.mu)
        out = (1 - self.mu) * x1 + self.mu * x2
        # out = x1 + x2
        return out


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, cylindrical=False):
        super().__init__()
        if not cylindrical:
            self.proj = nn.Conv3d(dim, dim_out, kernel_size=3, padding=1)
        else:
            self.proj = CylindricalConv(dim, dim_out, kernel_size=3, padding=1)
        try: 
            self.norm = nn.GroupNorm(groups, dim_out)
        except ValueError: 
            raise ValueError(f"Failed it init groupnorm with {groups} groups and {dim_out} out dims")
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    def __init__(self, dim, dim_out, *, cond_emb_dim=None, groups=8, cylindrical=False):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(cond_emb_dim, dim_out))
            if exists(cond_emb_dim)
            else None
        )

        conv = (
            CylindricalConv(dim, dim_out, kernel_size=1)
            if cylindrical
            else nn.Conv3d(dim, dim_out, kernel_size=1)
        )
        self.block1 = Block(dim, dim_out, groups=groups, cylindrical=cylindrical)
        self.block2 = Block(dim_out, dim_out, groups=groups, cylindrical=cylindrical)
        self.res_conv = conv if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            h = h + time_emb

        h = self.block2(h)
        return h + self.res_conv(x)


class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(
        self, dim, dim_out, *, cond_emb_dim=None, mult=2, norm=True, cylindrical=False
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(cond_emb_dim, dim))
            if exists(cond_emb_dim)
            else None
        )

        if not cylindrical:
            conv_op = nn.Conv3d
        else:
            conv_op = CylindricalConv

        self.ds_conv = conv_op(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            conv_op(dim, dim_out * mult, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            conv_op(dim_out * mult, dim_out, kernel_size=3, padding=1),
        )

        self.res_conv = (
            conv_op(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, cylindrical=False):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        debug = False

        if cylindrical:
            self.to_qkv = CylindricalConv(
                dim, hidden_dim * 3, kernel_size=1, bias=False
            )
            self.to_out = CylindricalConv(hidden_dim, dim, kernel_size=1)
        else:
            self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, kernel_size=1, bias=False)
            self.to_out = nn.Conv3d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, l, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y z) d -> b (h d) x y z", x=l, y=h, z=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=1, dim_head=32, cylindrical=False):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        if cylindrical:
            self.to_qkv = CylindricalConv(
                dim, hidden_dim * 3, kernel_size=1, bias=False
            )
            self.to_out = nn.Sequential(
                CylindricalConv(hidden_dim, dim, kernel_size=1), nn.GroupNorm(1, dim)
            )
        else:
            self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, kernel_size=1, bias=False)
            self.to_out = nn.Sequential(
                nn.Conv3d(hidden_dim, dim, kernel_size=1), nn.GroupNorm(1, dim)
            )

    def forward(self, x):
        b, c, l, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(
            out, "b h c (x y z) -> b (h c) x y z", h=self.heads, x=l, y=h, z=w
        )
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# up and down sample in 2 dims but keep z dimm


def Upsample(dim, extra_upsample=[0, 0, 0], cylindrical=False, compress_Z=False):
    Z_stride = 2 if compress_Z else 1
    Z_kernel = 4 if extra_upsample[0] > 0 else 3

    extra_upsample[0] = 0
    if cylindrical:
        return CylindricalConvTrans(
            dim,
            dim,
            kernel_size=(Z_kernel, 4, 4),
            stride=(Z_stride, 2, 2),
            padding=1,
            output_padding=extra_upsample,
        )
    else:
        return nn.ConvTranspose3d(
            dim,
            dim,
            kernel_size=(Z_kernel, 4, 4),
            stride=(Z_stride, 2, 2),
            padding=1,
            output_padding=extra_upsample,
        )


def Downsample(dim, cylindrical=False, compress_Z=False):
    Z_stride = 2 if compress_Z else 1
    if cylindrical:
        return CylindricalConv(
            dim, dim, kernel_size=(3, 4, 4), stride=(Z_stride, 2, 2), padding=1
        )
    else:
        return nn.Conv3d(
            dim, dim, kernel_size=(3, 4, 4), stride=(Z_stride, 2, 2), padding=1
        )
    # return nn.AvgPool3d(kernel_size = (1,2,2), stride = (1,2,2), padding =0)


class ResDense(nn.Module):
    # Single layer of dense resnet
    def __init__(self, dim, dim_out, cond_emb_dim=128):
        super().__init__()

        self.embeder = nn.Sequential(*[nn.GELU(), nn.Linear(cond_emb_dim, dim_out)])

        self.dense1 = nn.Sequential(*[nn.Linear(dim, dim_out), nn.GELU()])
        self.dense2 = nn.Sequential(*[nn.Linear(dim_out, dim_out), nn.GELU()])

    def forward(self, x, cond=None):
        h = self.dense1(x)
        embed = self.embeder(cond)
        h = h + embed
        h = self.dense2(h)

        return h + x


class ResNet(nn.Module):
    # Fully connected network with residual connection layers
    def __init__(
        self,
        dim_in=45,
        num_layers=3,
        hidden_dim=256,
        cond_emb_dim=128,
        cond_size=1,
    ):
        super().__init__()

        # time and energy embeddings
        half_cond_dim = cond_emb_dim // 2
        time_layers = []
        # if(time_embed): time_layers = [SinusoidalPositionEmbeddings(half_cond_dim//2)]
        time_layers = [
            nn.Unflatten(-1, (-1, 1)),
            nn.Linear(1, half_cond_dim // 2),
            nn.GELU(),
        ]
        time_layers += [
            nn.Linear(half_cond_dim // 2, half_cond_dim),
            nn.GELU(),
            nn.Linear(half_cond_dim, half_cond_dim),
        ]
        
        cond_layers = []
        # if(cond_embed): cond_layers = [SinusoidalPositionEmbeddings(half_cond_dim//2)]
        cond_layers = [nn.Linear(cond_size, half_cond_dim // 2), nn.GELU()]
        cond_layers += [
            nn.Linear(half_cond_dim // 2, half_cond_dim),
            nn.GELU(),
            nn.Linear(half_cond_dim, half_cond_dim),
        ]

        self.time_mlp = nn.Sequential(*time_layers)
        self.cond_mlp = nn.Sequential(*cond_layers)

        out_layers = [nn.Linear(dim_in + cond_emb_dim, dim_in)]

        # initial dense layer to hidden dim
        self.in_lay = nn.Linear(dim_in, hidden_dim)

        self.hidden_layers = nn.ModuleList([])
        # resnet for hidden dimmension
        for i in range(num_layers - 1):
            self.hidden_layers.append(
                ResDense(hidden_dim, hidden_dim, cond_emb_dim=cond_emb_dim)
            )

        # output
        self.out_lay = nn.Linear(hidden_dim, dim_in)

    def forward(self, x, cond=None, time=None, controls=None):
        c = self.cond_mlp(cond)
        t = self.time_mlp(time)
        cond = torch.cat([c, t], axis=-1)

        x = self.in_lay(x)
        for lay in self.hidden_layers:
            x = lay(x, cond)

        x = self.out_lay(x)

        return x


class FCN(nn.Module):
    # Fully connected network
    def __init__(
        self,
        dim_in=356,
        num_layers=4,
        cond_emb_dim=64,
        time_embed=True,
        cond_embed=True,
    ):
        super().__init__()

        # time and energy embeddings
        half_cond_dim = cond_emb_dim // 2
        time_layers = []
        if time_embed:
            time_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            time_layers = [
                nn.Unflatten(-1, (-1, 1)),
                nn.Linear(1, half_cond_dim // 2),
                nn.GELU(),
            ]
        time_layers += [
            nn.Linear(half_cond_dim // 2, half_cond_dim),
            nn.GELU(),
            nn.Linear(half_cond_dim, half_cond_dim),
        ]

        cond_layers = []
        if cond_embed:
            cond_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            cond_layers = [
                nn.Unflatten(-1, (-1, 1)),
                nn.Linear(1, half_cond_dim // 2),
                nn.GELU(),
            ]
        cond_layers += [
            nn.Linear(half_cond_dim // 2, half_cond_dim),
            nn.GELU(),
            nn.Linear(half_cond_dim, half_cond_dim),
        ]

        self.time_mlp = nn.Sequential(*time_layers)
        self.cond_mlp = nn.Sequential(*cond_layers)

        out_layers = [nn.Linear(dim_in + cond_emb_dim, dim_in)]
        for i in range(num_layers - 1):
            out_layers.append(nn.GELU())
            out_layers.append(nn.Linear(dim_in, dim_in))

        self.main_mlp = nn.Sequential(*out_layers)

    def forward(self, x, cond, time):
        t = self.time_mlp(time)
        c = self.cond_mlp(cond)
        x = torch.cat([x, t, c], axis=-1)

        x = self.main_mlp(x)
        return x


class CondUnet(nn.Module):
    # Unet with conditional layers
    def __init__(
        self,
        out_dim=1,
        layer_sizes=None,
        channels=1,
        cond_dim=128,
        resnet_block_groups=8,
        use_convnext=False,
        mid_attn=False,
        block_attn=False,
        compress_Z=False,
        convnext_mult=2,
        cylindrical=False,
        data_shape=(-1, 1, 45, 16, 9),
        time_embed=True,
        cond_embed=True,
        cond_size=1,
        no_time=False,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.block_attn = block_attn
        self.mid_attn = mid_attn

        self.no_time = no_time

        # dims = [channels, *map(lambda m: dim * m, dim_mults)]
        # layer_sizes.insert(0, channels)
        in_out = list(zip(layer_sizes[:-1], layer_sizes[1:]))

        if not cylindrical:
            self.init_conv = nn.Conv3d(
                channels, layer_sizes[0], kernel_size=3, padding=1
            )
        else:
            self.init_conv = CylindricalConv(
                channels, layer_sizes[0], kernel_size=3, padding=1
            )

        if use_convnext:
            block_klass = partial(
                ConvNextBlock, mult=convnext_mult, cylindrical=cylindrical
            )
        else:
            block_klass = partial(
                ResnetBlock, groups=resnet_block_groups, cylindrical=cylindrical
            )

        # time and energy embeddings
        half_cond_dim = cond_dim // 2

        time_layers = []
        if(not self.no_time):
            if time_embed:
                time_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
            else:
                time_layers = [
                    nn.Unflatten(-1, (-1, 1)),
                    nn.Linear(1, half_cond_dim // 2),
                    nn.GELU(),
                ]
            time_layers += [
                nn.Linear(half_cond_dim // 2, half_cond_dim),
                nn.GELU(),
                nn.Linear(half_cond_dim, half_cond_dim),
            ]
            self.time_mlp = nn.Sequential(*time_layers)

        last_cond_size = half_cond_dim if not self.no_time else cond_dim
        cond_layers = []
        cond_hidden_size = max(cond_size, half_cond_dim // 2)
        if cond_embed:
            cond_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            cond_layers = [nn.Linear(cond_size, cond_hidden_size), nn.GELU()]
        cond_layers += [
            nn.Linear(cond_hidden_size, half_cond_dim),
            nn.GELU(),
            nn.Linear(half_cond_dim, last_cond_size),
        ]

        self.cond_mlp = nn.Sequential(*cond_layers)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.downs_attn = nn.ModuleList([])
        self.ups_attn = nn.ModuleList([])
        self.extra_upsamples = []
        self.Z_even = []
        num_resolutions = len(in_out)

        cur_data_shape = data_shape[-3:]

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            if not is_last:
                extra_upsample_dim = [
                    (cur_data_shape[0] + 1) % 2,
                    cur_data_shape[1] % 2,
                    cur_data_shape[2] % 2,
                ]
                Z_dim = (
                    cur_data_shape[0]
                    if not compress_Z
                    else math.ceil(cur_data_shape[0] / 2.0)
                )
                cur_data_shape = (Z_dim, cur_data_shape[1] // 2, cur_data_shape[2] // 2)
                self.extra_upsamples.append(extra_upsample_dim)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, cond_emb_dim=cond_dim),
                        block_klass(dim_out, dim_out, cond_emb_dim=cond_dim),
                        Downsample(dim_out, cylindrical, compress_Z=compress_Z)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )
            if self.block_attn:
                self.downs_attn.append(
                    Residual(
                        PreNorm(
                            dim_out, LinearAttention(dim_out, cylindrical=cylindrical)
                        )
                    )
                )

        mid_dim = layer_sizes[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, cond_emb_dim=cond_dim)
        if self.mid_attn:
            self.mid_attn = Residual(
                PreNorm(mid_dim, LinearAttention(mid_dim, cylindrical=cylindrical))
            )
        self.mid_block2 = block_klass(mid_dim, mid_dim, cond_emb_dim=cond_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            if not is_last:
                extra_upsample = self.extra_upsamples.pop()

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, cond_emb_dim=cond_dim),
                        block_klass(dim_in, dim_in, cond_emb_dim=cond_dim),
                        Upsample(
                            dim_in, extra_upsample, cylindrical, compress_Z=compress_Z
                        )
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )
            if self.block_attn:
                self.ups_attn.append(
                    Residual(
                        PreNorm(
                            dim_in, LinearAttention(dim_in, cylindrical=cylindrical)
                        )
                    )
                )

        if not cylindrical:
            final_lay = nn.Conv3d(layer_sizes[0], out_dim, 1)
        else:
            final_lay = CylindricalConv(layer_sizes[0], out_dim, 1)
        self.final_conv = nn.Sequential(
            block_klass(layer_sizes[1], layer_sizes[0]), final_lay
        )

    def forward(self, x, cond=None, time=None, controls=None):
        x = self.init_conv(x)

        c = self.cond_mlp(cond)
        if(not self.no_time):
            t = self.time_mlp(time)
            conditions = torch.cat([t, c], axis=-1)
        else: 
            conditions = c

        h = []

        # downsample
        for i, (block1, block2, downsample) in enumerate(self.downs):
            x = block1(x, conditions)
            x = block2(x, conditions)
            if self.block_attn:
                x = self.downs_attn[i](x)
            h.append(x)
            x = downsample(x)

        # Add hidden state from controlnet
        if controls is not None:
            for i in range(len(h)):
                add_fn, control_h = controls[i]
                h[i] = add_fn(h[i], control_h)

        # bottleneck
        x = self.mid_block1(x, conditions)
        if self.mid_attn:
            x = self.mid_attn(x)
        x = self.mid_block2(x, conditions)

        # Add hidden state from controlnet
        if controls is not None:
            add_fn, control_h = controls[-1]
            x = add_fn(x, control_h)

        # upsample
        for i, (block1, block2, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, conditions)
            x = block2(x, conditions)
            if self.block_attn:
                x = self.ups_attn[i](x)
            x = upsample(x)

        return self.final_conv(x)

    def get_hiddens(self, x, cond, time):
        # Get list of hidden states of controlnet
        x = self.init_conv(x)

        t = self.time_mlp(time)
        c = self.cond_mlp(cond)

        conditions = torch.cat([t, c], axis=-1)

        hs = []

        # downsample
        for i, (block1, block2, downsample) in enumerate(self.downs):
            x = block1(x, conditions)
            x = block2(x, conditions)
            if self.block_attn:
                x = self.downs_attn[i](x)
            hs.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, conditions)
        if self.mid_attn:
            x = self.mid_attn(x)
        x = self.mid_block2(x, conditions)
        hs.append(x)

        return hs


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device = None):
        """Gated Root Mean Square Layer Normalization

        Paper: https://arxiv.org/abs/1910.07467
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise.

    Define this manually since torch's version doesn't seem to work on MPS.
    """
    return x * F.sigmoid(x)


class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """

    def __init__(
            self,
            img_size: list = [30,30,30],
            patch_size: list = [3,3,3],
            padding: list = [0, 0, 0],
            in_chans: int = 1,
            embed_dim: int = 1024,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.padding=padding
        self.img_size = img_size
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])   
        self.left_over = tuple([((g + 1) * p - s) % p for p, s, g in zip(self.patch_size, self.img_size, self.grid_size)]) 
        self.grid_size = tuple([(s + l ) // p for p, l, s in zip(self.patch_size, self.left_over, self.img_size)]) 
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2] #3D

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=padding, bias=bias) ## add posibility for zero padding

    def forward(self, x):
        #print(print("PatchEmbed layers: x input size", x.size()))
        B, C, R, PHI, Z  = x.shape
        x = F.pad(x, (0, self.left_over[2], 0, self.left_over[1], 0, self.left_over[0])) #add padding if needed (padding is inversed)
        #print(print("PatchEmbed layers: pre-proj", x.size()))
        x = self.proj(x)
        #print("PatchEmbed layers: post-proj", x.size())
        x = x.flatten(2).transpose(1, 2)  # NC Z PHI R -> NLC (N, 704, 1024)
        return x


class AttentionDiT(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = [bias, bias]
        drop_probs = [drop, drop]
        # linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        #linear_layer = nn.Linear

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        #self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):

    
        x = self.fc1(x)


        x = self.act(x)


        x = self.drop1(x)


        # Uncomment this if norm should be applied
        # x = self.norm(x)
        # print(f"After norm: {x.shape}, any NaNs: {torch.isnan(x).any()}")

        x = self.fc2(x)


        x = self.drop2(x)


        #if x is None:
        #    raise RuntimeError("MLP forward returned None!")
            
        return x

        
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, time_emb_dim=None, **block_kwargs):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionDiT(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlps = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 6 * hidden_size, bias=True)
        )

        

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(self.modulate(self.norm1(x), shift_msa, scale_msa))
        
        x_modulated = self.modulate(self.norm2(x), shift_mlp, scale_mlp)
   
        
        x = x + gate_mlp.unsqueeze(1) * self.mlps(x_modulated)
        return x

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    
    
def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

    
    
class PureDiT(nn.Module):
    def __init__(self, in_dim, dropout_ff=0.0, depth=4, hidden_dim=64, num_heads=8, mlp_ratio=4, time_embed = True,
            cond_embed = True, patch_shape = [30,30,30]):
        super(PureDiT, self).__init__()
        #self.config = config


        self.depth = depth
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        #self.cls_dim = config.cls_dim

        device = torch.device('cuda:0')

        self.out_dim = in_dim    



        self.drop_out = nn.Dropout(0.1) # else nn.Dropout(0)
                
        half_cond_dim = hidden_dim // 2
        time_layers = []
        if(time_embed): time_layers = [SinusoidalPositionEmbeddings(half_cond_dim//2)]
        else: time_layers = [nn.Unflatten(-1, (-1, 1)), nn.Linear(1, half_cond_dim//2),nn.GELU() ]
        time_layers += [ nn.Linear(half_cond_dim//2, half_cond_dim), nn.GELU(), nn.Linear(half_cond_dim, half_cond_dim)]


        cond_layers = []
        if(cond_embed): cond_layers = [SinusoidalPositionEmbeddings(half_cond_dim//2)]
        else: cond_layers = [nn.Unflatten(-1, (-1, 1)), nn.Linear(1, half_cond_dim//2),nn.GELU()]
        cond_layers += [ nn.Linear(half_cond_dim//2, half_cond_dim), nn.GELU(), nn.Linear(half_cond_dim, half_cond_dim)]


        self.time_mlp = nn.Sequential(*time_layers)
        self.cond_mlp = nn.Sequential(*cond_layers)


        out_layers = [nn.Linear(in_dim + hidden_dim, in_dim)]


        self.embeder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )


        input_size = patch_shape
        #input_size = [45,50,18]
        patch_size = [3, 3, 3]  ## Note stride = patch
        padding = [0, 0, 0]
        self.patch_embedder = PatchEmbed(input_size, patch_size, padding, in_dim, hidden_dim, bias=True)
        num_patches = self.patch_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, self.num_heads, mlp_ratio=self.mlp_ratio, time_emb_dim=hidden_dim ) for _ in range(self.depth)
        ])

        
        
        self.out_channels = 1
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_dim, patch_size[0] * patch_size[1] * patch_size[2] * self.out_channels, bias=True),
        )
        
        
        z_dims = [32]
        
        self.projectors = nn.ModuleList([
            build_mlp(hidden_dim, hidden_dim, z_dim) for z_dim in z_dims
            ])
        
        

        
        
    def unpatchify(self, x, channels):
        """
        input: (N, T, patch_size[0] * patch_size[1] * patch_size[2] * C)    (N, 704, 2*2*2*1)
        voxels: (N, C, Z, PHI, R)          (N, 1, 45, 16, 9)
        """
        c = channels
        p_r = self.patch_embedder.patch_size[0]
        p_phi = self.patch_embedder.patch_size[1]
        p_z = self.patch_embedder.patch_size[2]
        r_lo = self.patch_embedder.left_over[0]
        phi_lo = self.patch_embedder.left_over[1]
        z_lo = self.patch_embedder.left_over[2]
        r = self.patch_embedder.grid_size[0]
        phi = self.patch_embedder.grid_size[1]
        z = self.patch_embedder.grid_size[2]
        assert r * phi * z == x.shape[1]

        x = x.reshape(shape=(x.shape[0], r, phi, z, p_r, p_phi, p_z, c))
        x = torch.einsum('nhwdpqrc->nchpwqdr', x)
        imgs = x.reshape(shape=(x.shape[0], c, r * p_r, phi * p_phi, z  * p_z))

        imgs = imgs[:,:,:r * p_r - r_lo, :phi * p_phi - phi_lo, :z * p_z - z_lo]
        return imgs



    def forward(self, data, cond=None, time=None, return_patch=False, encoder_patch=False):
        
        

        t = self.time_mlp(time)
        c = self.cond_mlp(cond)
        #c1 = self.cond_mlp(cond)
        conditionss = torch.cat([c, t], axis=-1)
        
        
        
        #data size (B,C,R,Z,PHI)
        
        x = self.patch_embedder(data) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        second_layer_patch = None
        
        N, T, D = x.shape
        
        
        for idx, block in enumerate(self.blocks):
            x = block(x, conditionss)                    # (B, T, D)

            # stash the output after the 2nd layer (idx == 1)
            if encoder_patch and idx == 1:
                # keep exact tensor (B, T, D)
                second_layer_patch = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]

        # --- patch-return paths ---
        if return_patch:
            if encoder_patch:
                # REPA path: give 3 things in fixed order
                # 1) final diffusion patch
                # 2) early / 2nd-layer patch
                # 3) encoder patch (external)
                return x, second_layer_patch
            else:
                # classic DINO/diffu-style patch read
                return x

        else:
            #print(f'after the block {x.shape}')
            x = self.final_layer(x)                     # (N, T, patch_size ** 2 * out_channels)
            #print(f'x after final layer {x.shape}')
            preds = self.unpatchify(x, self.out_channels)   # (N, out_channels, H, W)

            #print(f'x after unpatchify {preds.shape}')

            if encoder_patch:
                return preds, second_layer_patch
            else:
                return preds
    

class MeanFlowDiT(nn.Module):
    def __init__(self, in_dim, dropout_ff=0.0, depth=4, hidden_dim=96, num_heads=8, mlp_ratio=4, time_embed = True,
            cond_embed = True, patch_shape = [30,30,30]):
        super(MeanFlowDiT, self).__init__()
        #self.config = config


        self.depth = depth
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        #self.cls_dim = config.cls_dim

        device = torch.device('cuda:0')

        self.out_dim = in_dim    



        self.drop_out = nn.Dropout(0.1) # else nn.Dropout(0)
                
        # ---- sizing (keep your style) ----
        #half_cond_dim = hidden_dim // 3
        half_cond_dim = hidden_dim
        #assert half_cond_dim > 0, "hidden_dim must be >= 3"

        # ---- time branch ----
        time_layers = []
        if (time_embed):
            time_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            time_layers = [nn.Unflatten(-1, (-1, 1)), nn.Linear(1, half_cond_dim // 2), nn.GELU()]
        time_layers += [nn.Linear(half_cond_dim // 2, half_cond_dim), nn.GELU(), nn.Linear(half_cond_dim, half_cond_dim)]
        self.time_layers = nn.Sequential(*time_layers)

        # ---- cond branch ----
        cond_layers = []
        if (cond_embed):
            cond_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            cond_layers = [nn.Unflatten(-1, (-1, 1)), nn.Linear(1, half_cond_dim // 2), nn.GELU()]
        cond_layers += [nn.Linear(half_cond_dim // 2, half_cond_dim), nn.GELU(), nn.Linear(half_cond_dim, half_cond_dim)]
        self.cond_layers = nn.Sequential(*cond_layers)

        # ---- r (delta) branch: NEW ----
        r_layers = []
        if (time_embed):  # mirror "time" branch choice
            r_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            r_layers = [nn.Unflatten(-1, (-1, 1)), nn.Linear(1, half_cond_dim // 2), nn.GELU()]
        r_layers += [nn.Linear(half_cond_dim // 2, half_cond_dim), nn.GELU(), nn.Linear(half_cond_dim, half_cond_dim)]
        self.r_layers = nn.Sequential(*r_layers)

        out_layers = [nn.Linear(in_dim + hidden_dim, in_dim)]


        self.embeder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )


        input_size = patch_shape
        #input_size = [45,50,18]
        patch_size = [3, 3, 3]  ## Note stride = patch
        padding = [0, 0, 0]
        self.patch_embedder = PatchEmbed(input_size, patch_size, padding, in_dim, hidden_dim, bias=True)
        num_patches = self.patch_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, self.num_heads, mlp_ratio=self.mlp_ratio, time_emb_dim=hidden_dim ) for _ in range(self.depth)
        ])

        
        
        self.out_channels = 1
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_dim, patch_size[0] * patch_size[1] * patch_size[2] * self.out_channels, bias=True),
        )
        
        
        z_dims = [32]
        
        self.projectors = nn.ModuleList([
            build_mlp(hidden_dim, hidden_dim, z_dim) for z_dim in z_dims
            ])
        
        

        
        
    def unpatchify(self, x, channels):
        """
        input: (N, T, patch_size[0] * patch_size[1] * patch_size[2] * C)    (N, 704, 2*2*2*1)
        voxels: (N, C, Z, PHI, R)          (N, 1, 45, 16, 9)
        """
        c = channels
        p_r = self.patch_embedder.patch_size[0]
        p_phi = self.patch_embedder.patch_size[1]
        p_z = self.patch_embedder.patch_size[2]
        r_lo = self.patch_embedder.left_over[0]
        phi_lo = self.patch_embedder.left_over[1]
        z_lo = self.patch_embedder.left_over[2]
        r = self.patch_embedder.grid_size[0]
        phi = self.patch_embedder.grid_size[1]
        z = self.patch_embedder.grid_size[2]
        assert r * phi * z == x.shape[1]

        x = x.reshape(shape=(x.shape[0], r, phi, z, p_r, p_phi, p_z, c))
        x = torch.einsum('nhwdpqrc->nchpwqdr', x)
        imgs = x.reshape(shape=(x.shape[0], c, r * p_r, phi * p_phi, z  * p_z))

        imgs = imgs[:,:,:r * p_r - r_lo, :phi * p_phi - phi_lo, :z * p_z - z_lo]
        return imgs



    def forward(self, data, cond=None, time=None, r=None):
        
        

        t = self.time_layers(time)
        c = self.cond_layers(cond)
        r_embed = self.r_layers(time-r) #### v2
        #r_embed = self.r_layers(r) 
        #conditionss = torch.cat([c, t, r_embed], axis=-1)
        
        conditionss = c + t + r_embed
        
        

        
        x = self.patch_embedder(data) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        second_layer_patch = None
        
        N, T, D = x.shape
        
        
        for idx, block in enumerate(self.blocks):
            x = block(x, conditionss)                    # (B, T, D)


        #print(f'after the block {x.shape}')
        x = self.final_layer(x)                     # (N, T, patch_size ** 2 * out_channels)
        #print(f'x after final layer {x.shape}')
        preds = self.unpatchify(x, self.out_channels)   # (N, out_channels, H, W)

        return preds

class MeanFlowDiT_v1(nn.Module):
    def __init__(self, in_dim, dropout_ff=0.0, depth=4, hidden_dim=96, num_heads=8, mlp_ratio=4, time_embed = True,
            cond_embed = True, patch_shape = [30,30,30]):
        super(MeanFlowDiT_v1, self).__init__()
        #self.config = config


        self.depth = depth
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        #self.cls_dim = config.cls_dim

        device = torch.device('cuda:0')

        self.out_dim = in_dim    



        self.drop_out = nn.Dropout(0.1) # else nn.Dropout(0)
                
        # ---- sizing (keep your style) ----
        #half_cond_dim = hidden_dim // 3
        half_cond_dim = hidden_dim
        #assert half_cond_dim > 0, "hidden_dim must be >= 3"

        # ---- time branch ----
        time_layers = []
        if (time_embed):
            time_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            time_layers = [nn.Unflatten(-1, (-1, 1)), nn.Linear(1, half_cond_dim // 2), nn.GELU()]
        time_layers += [nn.Linear(half_cond_dim // 2, half_cond_dim), nn.GELU(), nn.Linear(half_cond_dim, half_cond_dim)]
        self.time_layers = nn.Sequential(*time_layers)

        # ---- cond branch ----
        cond_layers = []
        if (cond_embed):
            cond_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            cond_layers = [nn.Unflatten(-1, (-1, 1)), nn.Linear(1, half_cond_dim // 2), nn.GELU()]
        cond_layers += [nn.Linear(half_cond_dim // 2, half_cond_dim), nn.GELU(), nn.Linear(half_cond_dim, half_cond_dim)]
        self.cond_layers = nn.Sequential(*cond_layers)

        # ---- r (delta) branch: NEW ----
        r_layers = []
        if (time_embed):  # mirror "time" branch choice
            r_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            r_layers = [nn.Unflatten(-1, (-1, 1)), nn.Linear(1, half_cond_dim // 2), nn.GELU()]
        r_layers += [nn.Linear(half_cond_dim // 2, half_cond_dim), nn.GELU(), nn.Linear(half_cond_dim, half_cond_dim)]
        self.r_layers = nn.Sequential(*r_layers)

        out_layers = [nn.Linear(in_dim + hidden_dim, in_dim)]


        self.embeder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )


        input_size = patch_shape
        #input_size = [45,50,18]
        patch_size = [3, 3, 3]  ## Note stride = patch
        padding = [0, 0, 0]
        self.patch_embedder = PatchEmbed(input_size, patch_size, padding, in_dim, hidden_dim, bias=True)
        num_patches = self.patch_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, self.num_heads, mlp_ratio=self.mlp_ratio, time_emb_dim=hidden_dim ) for _ in range(self.depth)
        ])

        
        
        self.out_channels = 1
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_dim, patch_size[0] * patch_size[1] * patch_size[2] * self.out_channels, bias=True),
        )
        
        
        z_dims = [32]
        
        self.projectors = nn.ModuleList([
            build_mlp(hidden_dim, hidden_dim, z_dim) for z_dim in z_dims
            ])
        
        

        
        
    def unpatchify(self, x, channels):
        """
        input: (N, T, patch_size[0] * patch_size[1] * patch_size[2] * C)    (N, 704, 2*2*2*1)
        voxels: (N, C, Z, PHI, R)          (N, 1, 45, 16, 9)
        """
        c = channels
        p_r = self.patch_embedder.patch_size[0]
        p_phi = self.patch_embedder.patch_size[1]
        p_z = self.patch_embedder.patch_size[2]
        r_lo = self.patch_embedder.left_over[0]
        phi_lo = self.patch_embedder.left_over[1]
        z_lo = self.patch_embedder.left_over[2]
        r = self.patch_embedder.grid_size[0]
        phi = self.patch_embedder.grid_size[1]
        z = self.patch_embedder.grid_size[2]
        assert r * phi * z == x.shape[1]

        x = x.reshape(shape=(x.shape[0], r, phi, z, p_r, p_phi, p_z, c))
        x = torch.einsum('nhwdpqrc->nchpwqdr', x)
        imgs = x.reshape(shape=(x.shape[0], c, r * p_r, phi * p_phi, z  * p_z))

        imgs = imgs[:,:,:r * p_r - r_lo, :phi * p_phi - phi_lo, :z * p_z - z_lo]
        return imgs



    def forward(self, data, cond=None, time=None, r=None):
        
        

        t = self.time_layers(time)
        c = self.cond_layers(cond)
        r_embed = self.r_layers(r) #### v1

        
        conditionss = c + t + r_embed
        
        

        
        x = self.patch_embedder(data) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        second_layer_patch = None
        
        N, T, D = x.shape
        
        
        for idx, block in enumerate(self.blocks):
            x = block(x, conditionss)                    # (B, T, D)


        #print(f'after the block {x.shape}')
        x = self.final_layer(x)                     # (N, T, patch_size ** 2 * out_channels)
        #print(f'x after final layer {x.shape}')
        preds = self.unpatchify(x, self.out_channels)   # (N, out_channels, H, W)

        return preds    



class MeanFlowCondUnet_v2(nn.Module):
#Unet with conditional layers
    def __init__(
        self,
        out_dim=1,
        layer_sizes = None,
        channels=1,
        cond_dim = 64,
        resnet_block_groups=8,
        use_convnext=False,
        mid_attn = False,
        block_attn = False,
        compress_Z = False,
        convnext_mult=2,
        cylindrical = False,
        data_shape = (-1,1,45, 16,9),
        time_embed = True,
        cond_embed = True,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.block_attn = block_attn
        self.mid_attn = mid_attn



        #dims = [channels, *map(lambda m: dim * m, dim_mults)]
        #layer_sizes.insert(0, channels)
        in_out = list(zip(layer_sizes[:-1], layer_sizes[1:])) 
        
        if(not cylindrical): self.init_conv = nn.Conv3d(channels, layer_sizes[0], kernel_size = 3, padding = 1)
        else: self.init_conv = CylindricalConv(channels, layer_sizes[0], kernel_size = 3, padding = 1)

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult, cylindrical = cylindrical)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups, cylindrical = cylindrical)

        # ---- sizing (keep your style) ----
        #half_cond_dim = hidden_dim // 3
        half_cond_dim = cond_dim
        #assert half_cond_dim > 0, "hidden_dim must be >= 3"

        # ---- time branch ----
        time_layers = []
        if (time_embed):
            time_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            time_layers = [nn.Unflatten(-1, (-1, 1)), nn.Linear(1, half_cond_dim // 2), nn.GELU()]
        time_layers += [nn.Linear(half_cond_dim // 2, half_cond_dim), nn.GELU(), nn.Linear(half_cond_dim, half_cond_dim)]
        self.time_layers = nn.Sequential(*time_layers)

        # ---- cond branch ----
        cond_layers = []
        if (cond_embed):
            cond_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            cond_layers = [nn.Unflatten(-1, (-1, 1)), nn.Linear(1, half_cond_dim // 2), nn.GELU()]
        cond_layers += [nn.Linear(half_cond_dim // 2, half_cond_dim), nn.GELU(), nn.Linear(half_cond_dim, half_cond_dim)]
        self.cond_layers = nn.Sequential(*cond_layers)

        # ---- r (delta) branch: NEW ----
        r_layers = []
        if (time_embed):  # mirror "time" branch choice
            r_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            r_layers = [nn.Unflatten(-1, (-1, 1)), nn.Linear(1, half_cond_dim // 2), nn.GELU()]
        r_layers += [nn.Linear(half_cond_dim // 2, half_cond_dim), nn.GELU(), nn.Linear(half_cond_dim, half_cond_dim)]
        self.r_layers = nn.Sequential(*r_layers)


        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.downs_attn = nn.ModuleList([])
        self.ups_attn = nn.ModuleList([])
        self.extra_upsamples = []
        self.Z_even = []
        num_resolutions = len(in_out)

        cur_data_shape = data_shape[-3:]
        
        #print(f'data_shape {data_shape}')
        #print(f'cur_data_shape {cur_data_shape}')

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = (ind >= (num_resolutions - 1))
            if(not is_last):
                if compress_Z:
                    extra_z = (cur_data_shape[0] + 1)%2
                else:
                    extra_z = 0  # depth stays fixed, so no correction

                extra_upsample_dim = [extra_z,
                                      cur_data_shape[1] % 2,
                                      cur_data_shape[2] % 2]

                Z_dim = cur_data_shape[0] if not compress_Z else math.ceil(cur_data_shape[0]/2.0)
                cur_data_shape = (Z_dim, cur_data_shape[1] // 2, cur_data_shape[2] //2)
                self.extra_upsamples.append(extra_upsample_dim)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, cond_emb_dim=cond_dim),
                        block_klass(dim_out, dim_out, cond_emb_dim=cond_dim),
                        Downsample(dim_out, cylindrical, compress_Z = compress_Z) if not is_last else nn.Identity(),
                    ]
                )
            )
            if(self.block_attn) : self.downs_attn.append(Residual(PreNorm(dim_out, LinearAttention(dim_out, cylindrical = cylindrical))))

        mid_dim = layer_sizes[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, cond_emb_dim=cond_dim)
        if(self.mid_attn): self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim, cylindrical = cylindrical)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, cond_emb_dim=cond_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = (ind >= (num_resolutions - 1))

            if(not is_last): 
                extra_upsample = self.extra_upsamples.pop()


            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, cond_emb_dim=cond_dim),
                        block_klass(dim_in, dim_in, cond_emb_dim=cond_dim),
                        Upsample(dim_in, extra_upsample, cylindrical, compress_Z = compress_Z) if not is_last else nn.Identity(),
                    ]
                )
            )
            if(self.block_attn): self.ups_attn.append( Residual(PreNorm(dim_in, LinearAttention(dim_in, cylindrical = cylindrical))) )

        if(not cylindrical): final_lay = nn.Conv3d(layer_sizes[0], out_dim, 1)
        else:  final_lay = CylindricalConv(layer_sizes[0], out_dim, 1)
        self.final_conv = nn.Sequential( block_klass(layer_sizes[1], layer_sizes[0]),  final_lay )

    def forward(self, x, cond=None, time=None, r=None):

        x = self.init_conv(x)

        t = self.time_layers(time)
        c = self.cond_layers(cond)
        r_embed = self.r_layers(time-r) #### v2
        
        conditions = c + t + r_embed


        h = []

        # downsample
        for i, (block1, block2, downsample) in enumerate(self.downs):
            x = block1(x, conditions)
            x = block2(x, conditions)
            if(self.block_attn): x = self.downs_attn[i](x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, conditions)
        if(self.mid_attn): x = self.mid_attn(x)
        x = self.mid_block2(x, conditions)
        


        # upsample
        for i, (block1, block2, upsample) in enumerate(self.ups):
            skip = h.pop()
            #print(f"UP {i}: x {x.shape}, skip {skip.shape}")  # <--- NEW

            x = torch.cat((x, skip), dim=1)
            x = block1(x, conditions)
            x = block2(x, conditions)
            if(self.block_attn): x = self.ups_attn[i](x)
            x = upsample(x)

        return self.final_conv(x)
    
    
class MeanFlowCondUnet(nn.Module):
#Unet with conditional layers
    def __init__(
        self,
        out_dim=1,
        layer_sizes = None,
        channels=1,
        cond_dim = 64,
        resnet_block_groups=8,
        use_convnext=False,
        mid_attn = False,
        block_attn = False,
        compress_Z = False,
        convnext_mult=2,
        cylindrical = False,
        data_shape = (-1,1,45, 16,9),
        time_embed = True,
        cond_embed = True,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.block_attn = block_attn
        self.mid_attn = mid_attn



        #dims = [channels, *map(lambda m: dim * m, dim_mults)]
        #layer_sizes.insert(0, channels)
        in_out = list(zip(layer_sizes[:-1], layer_sizes[1:])) 
        
        if(not cylindrical): self.init_conv = nn.Conv3d(channels, layer_sizes[0], kernel_size = 3, padding = 1)
        else: self.init_conv = CylindricalConv(channels, layer_sizes[0], kernel_size = 3, padding = 1)

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult, cylindrical = cylindrical)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups, cylindrical = cylindrical)

        # ---- sizing (keep your style) ----
        #half_cond_dim = hidden_dim // 3
        half_cond_dim = cond_dim
        #assert half_cond_dim > 0, "hidden_dim must be >= 3"

        # ---- time branch ----
        time_layers = []
        if (time_embed):
            time_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            time_layers = [nn.Unflatten(-1, (-1, 1)), nn.Linear(1, half_cond_dim // 2), nn.GELU()]
        time_layers += [nn.Linear(half_cond_dim // 2, half_cond_dim), nn.GELU(), nn.Linear(half_cond_dim, half_cond_dim)]
        self.time_layers = nn.Sequential(*time_layers)

        # ---- cond branch ----
        cond_layers = []
        if (cond_embed):
            cond_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            cond_layers = [nn.Unflatten(-1, (-1, 1)), nn.Linear(1, half_cond_dim // 2), nn.GELU()]
        cond_layers += [nn.Linear(half_cond_dim // 2, half_cond_dim), nn.GELU(), nn.Linear(half_cond_dim, half_cond_dim)]
        self.cond_layers = nn.Sequential(*cond_layers)

        # ---- r (delta) branch: NEW ----
        r_layers = []
        if (time_embed):  # mirror "time" branch choice
            r_layers = [SinusoidalPositionEmbeddings(half_cond_dim // 2)]
        else:
            r_layers = [nn.Unflatten(-1, (-1, 1)), nn.Linear(1, half_cond_dim // 2), nn.GELU()]
        r_layers += [nn.Linear(half_cond_dim // 2, half_cond_dim), nn.GELU(), nn.Linear(half_cond_dim, half_cond_dim)]
        self.r_layers = nn.Sequential(*r_layers)


        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.downs_attn = nn.ModuleList([])
        self.ups_attn = nn.ModuleList([])
        self.extra_upsamples = []
        self.Z_even = []
        num_resolutions = len(in_out)

        cur_data_shape = data_shape[-3:]
        
        #print(f'data_shape {data_shape}')
        #print(f'cur_data_shape {cur_data_shape}')

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = (ind >= (num_resolutions - 1))
            if(not is_last):
                if compress_Z:
                    extra_z = (cur_data_shape[0] + 1)%2
                else:
                    extra_z = 0  # depth stays fixed, so no correction

                extra_upsample_dim = [extra_z,
                                      cur_data_shape[1] % 2,
                                      cur_data_shape[2] % 2]

                Z_dim = cur_data_shape[0] if not compress_Z else math.ceil(cur_data_shape[0]/2.0)
                cur_data_shape = (Z_dim, cur_data_shape[1] // 2, cur_data_shape[2] //2)
                self.extra_upsamples.append(extra_upsample_dim)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, cond_emb_dim=cond_dim),
                        block_klass(dim_out, dim_out, cond_emb_dim=cond_dim),
                        Downsample(dim_out, cylindrical, compress_Z = compress_Z) if not is_last else nn.Identity(),
                    ]
                )
            )
            if(self.block_attn) : self.downs_attn.append(Residual(PreNorm(dim_out, LinearAttention(dim_out, cylindrical = cylindrical))))

        mid_dim = layer_sizes[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, cond_emb_dim=cond_dim)
        if(self.mid_attn): self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim, cylindrical = cylindrical)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, cond_emb_dim=cond_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = (ind >= (num_resolutions - 1))

            if(not is_last): 
                extra_upsample = self.extra_upsamples.pop()


            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, cond_emb_dim=cond_dim),
                        block_klass(dim_in, dim_in, cond_emb_dim=cond_dim),
                        Upsample(dim_in, extra_upsample, cylindrical, compress_Z = compress_Z) if not is_last else nn.Identity(),
                    ]
                )
            )
            if(self.block_attn): self.ups_attn.append( Residual(PreNorm(dim_in, LinearAttention(dim_in, cylindrical = cylindrical))) )

        if(not cylindrical): final_lay = nn.Conv3d(layer_sizes[0], out_dim, 1)
        else:  final_lay = CylindricalConv(layer_sizes[0], out_dim, 1)
        self.final_conv = nn.Sequential( block_klass(layer_sizes[1], layer_sizes[0]),  final_lay )

    def forward(self, x, cond=None, time=None, r=None):

        x = self.init_conv(x)

        t = self.time_layers(time)
        c = self.cond_layers(cond)
        r_embed = self.r_layers(r) #### v1
        
        conditions = c + t + r_embed


        h = []

        # downsample
        for i, (block1, block2, downsample) in enumerate(self.downs):
            x = block1(x, conditions)
            x = block2(x, conditions)
            if(self.block_attn): x = self.downs_attn[i](x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, conditions)
        if(self.mid_attn): x = self.mid_attn(x)
        x = self.mid_block2(x, conditions)
        


        # upsample
        for i, (block1, block2, upsample) in enumerate(self.ups):
            skip = h.pop()
            #print(f"UP {i}: x {x.shape}, skip {skip.shape}")  # <--- NEW

            x = torch.cat((x, skip), dim=1)
            x = block1(x, conditions)
            x = block2(x, conditions)
            if(self.block_attn): x = self.ups_attn[i](x)
            x = upsample(x)

        return self.final_conv(x)
