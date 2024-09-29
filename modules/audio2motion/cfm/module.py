import torch
from torch import nn, Tensor, einsum, IntTensor, FloatTensor, BoolTensor
from torch.nn import Module
import torch.nn.functional as F

from beartype import beartype
from beartype.typing import Tuple, Optional, List, Union

from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, pack, unpack

# from gateloop_transformer import SimpleGateLoopLayer as GateLoop

from modules.audio2motion.cfm.utils import *
from modules.audio2motion.cfm.attend import Attend

import math
from functools import partial
from torch.cuda.amp import autocast

# sinusoidal positions

class LearnedSinusoidalPosEmb(Module):
    """ used by @crowsonkb """

    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        return fouriered

# rotary positional embeddings
# https://arxiv.org/abs/2104.09864

class RotaryEmbedding(Module):
    def __init__(self, dim, theta = 50000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @autocast(enabled = False)
    @beartype
    def forward(self, t: Union[int, Tensor]):
        if not torch.is_tensor(t):
            t = torch.arange(t, device = self.device)

        t = t.type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)

@autocast(enabled = False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()

# convolutional positional generating module

class ConvPositionEmbed(Module):
    def __init__(
        self,
        dim,
        *,
        kernel_size,
        groups = None
    ):
        super().__init__()
        assert is_odd(kernel_size)
        groups = default(groups, dim) # full depthwise conv by default

        self.dw_conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups = groups, padding = kernel_size // 2),
            nn.GELU()
        )

    def forward(self, x):
        x = rearrange(x, 'b n c -> b c n')
        x = self.dw_conv1d(x)
        return rearrange(x, 'b c n -> b n c')

# norms

class RMSNorm(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        cond_dim = None
    ):
        super().__init__()
        cond_dim = default(cond_dim, dim)
        self.scale = dim ** 0.5

        self.to_gamma = nn.Linear(cond_dim, dim)
        self.to_beta = nn.Linear(cond_dim, dim)

        # init to identity

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)

        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, *, cond):
        normed = F.normalize(x, dim = -1) * self.scale

        gamma, beta = self.to_gamma(cond), self.to_beta(cond)
        gamma, beta = map(lambda t: rearrange(t, 'b d -> b 1 d'), (gamma, beta))

        return normed * gamma + beta

# attention

class MultiheadRMSNorm(Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0,
        flash = False,
        qk_norm = False,
        qk_norm_scale = 10
    ):
        super().__init__()
        self.heads = heads
        dim_inner = dim_head * heads

        scale = qk_norm_scale if qk_norm else None

        self.attend = Attend(dropout, flash = flash, scale = scale)

        self.qk_norm = qk_norm

        if qk_norm:
            self.q_norm = MultiheadRMSNorm(dim_head, heads = heads)
            self.k_norm = MultiheadRMSNorm(dim_head, heads = heads)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(self, x, mask = None, rotary_emb = None):
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if exists(rotary_emb):
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# feedforward

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim)
    )
    

# transformer
class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        num_register_tokens = 0.,
        attn_flash = False,
        adaptive_rmsnorm = False,
        adaptive_rmsnorm_cond_dim_in = None,
        use_unet_skip_connection = False,
        skip_connect_scale = None,
        attn_qk_norm = False,
        use_gateloop_layers = False
    ):
        super().__init__()
        assert divisible_by(depth, 2)
        self.layers = nn.ModuleList([])

        self.rotary_emb = RotaryEmbedding(dim = dim_head)

        self.num_register_tokens = num_register_tokens
        self.has_register_tokens = num_register_tokens > 0

        if self.has_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        if adaptive_rmsnorm:
            rmsnorm_klass = partial(AdaptiveRMSNorm, cond_dim = adaptive_rmsnorm_cond_dim_in)
        else:
            rmsnorm_klass = RMSNorm

        self.skip_connect_scale = default(skip_connect_scale, 2 ** -0.5)

        for ind in range(depth):
            layer = ind + 1
            has_skip = use_unet_skip_connection and layer > (depth // 2)

            self.layers.append(nn.ModuleList([
                nn.Linear(dim * 2, dim) if has_skip else None,
                # GateLoop(dim = dim) if use_gateloop_layers else None,
                None,
                rmsnorm_klass(dim = dim),
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = attn_flash, qk_norm = attn_qk_norm),
                rmsnorm_klass(dim = dim),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.final_norm = RMSNorm(dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x,
        mask = None,
        adaptive_rmsnorm_cond = None
    ):
        batch, seq_len, *_ = x.shape

        # add register tokens to the left

        if self.has_register_tokens:
            register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = batch)

            x, ps = pack([register_tokens, x], 'b * d')

            if exists(mask):
                mask = F.pad(mask, (self.num_register_tokens, 0), value = True)

        # keep track of skip connections

        skip_connects = []

        # rotary embeddings

        positions = seq_len

        if self.has_register_tokens:
            main_positions = torch.arange(seq_len, device = self.device, dtype = torch.long)
            register_positions = torch.full((self.num_register_tokens,), -10000, device = self.device, dtype = torch.long)
            positions = torch.cat((register_positions, main_positions))

        rotary_emb = self.rotary_emb(positions)

        # adaptive rmsnorm

        rmsnorm_kwargs = dict()
        if exists(adaptive_rmsnorm_cond):
            rmsnorm_kwargs = dict(cond = adaptive_rmsnorm_cond)

        # going through the attention layers

        for skip_combiner, maybe_gateloop, attn_prenorm, attn, ff_prenorm, ff in self.layers:

            # in the paper, they use a u-net like skip connection
            # unclear how much this helps, as no ablations or further numbers given besides a brief one-two sentence mention

            if not exists(skip_combiner):
                skip_connects.append(x)
            else:
                skip_connect = skip_connects.pop() * self.skip_connect_scale
                x = torch.cat((x, skip_connect), dim = -1)
                x = skip_combiner(x)

            if exists(maybe_gateloop):
                x = maybe_gateloop(x) + x

            attn_input = attn_prenorm(x, **rmsnorm_kwargs)
            x = attn(attn_input, mask = mask, rotary_emb = rotary_emb) + x

            ff_input = ff_prenorm(x, **rmsnorm_kwargs) 
            x = ff(ff_input) + x

        # remove the register tokens

        if self.has_register_tokens:
            _, x = unpack(x, ps, 'b * d')

        return self.final_norm(x)

if __name__ == '__main__':
    # Initialize the Transformer
    transformer = Transformer(dim=512, depth=6, dim_head=64, heads=8, ff_mult=4)

    # Create random input tensor
    input_tensor = torch.randn(1, 10, 512)  # Assuming input shape is (batch_size, sequence_length, input_dim)

    # Forward pass through the Transformer
    output = transformer(input_tensor)

    # Print the shape of the output
    print(output.shape)

