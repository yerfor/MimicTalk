import torch
from torch import nn, Tensor, einsum, IntTensor, FloatTensor, BoolTensor
from torch.nn import Module
import torch.nn.functional as F
import random
from beartype import beartype
from beartype.typing import Tuple, Optional, List, Union

from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, pack, unpack

from modules.audio2motion.cfm.utils import exists, identity, default, divisible_by, is_odd, coin_flip, pack_one, unpack_one
from modules.audio2motion.cfm.utils import prob_mask_like, reduce_masks_with_and, interpolate_1d, curtail_or_pad, mask_from_start_end_indices, mask_from_frac_lengths
from modules.audio2motion.cfm.module import ConvPositionEmbed, LearnedSinusoidalPosEmb, Transformer
from torch.cuda.amp import autocast

class InContextTransformerAudio2Motion(Module):
    def __init__(
        self,
        *,
        dim_in = 64, # expression code
        dim_audio_in = 1024,
        dim = 1024,
        depth = 24,
        dim_head = 64,
        heads = 16,
        ff_mult = 4,
        ff_dropout = 0.,
        time_hidden_dim = None,
        conv_pos_embed_kernel_size = 31,
        conv_pos_embed_groups = None,
        attn_dropout = 0,
        attn_flash = False,
        attn_qk_norm = True,
        use_gateloop_layers = False,
        num_register_tokens = 16,
        frac_lengths_mask: Tuple[float, float] = (0.7, 1.),
    ):
        super().__init__()
        dim_in = default(dim_in, dim)

        time_hidden_dim = default(time_hidden_dim, dim * 4)

        self.proj_in = nn.Identity()
        self.sinu_pos_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim),
            nn.Linear(dim, time_hidden_dim),
            nn.SiLU()
        )

        self.dim_audio_in = dim_audio_in
        if self.dim_audio_in != dim_in:
            self.to_cond_emb = nn.Linear(self.dim_audio_in, dim_in)
        else:
            self.to_cond_emb = nn.Identity()

        # self.p_drop_prob = p_drop_prob
        self.frac_lengths_mask = frac_lengths_mask

        self.to_embed = nn.Linear(dim_in * 2 + dim_in, dim)

        self.null_cond = nn.Parameter(torch.zeros(dim_in))

        self.conv_embed = ConvPositionEmbed(
            dim = dim,
            kernel_size = conv_pos_embed_kernel_size,
            groups = conv_pos_embed_groups
        )

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            ff_mult = ff_mult,
            ff_dropout = ff_dropout,
            attn_dropout= attn_dropout,
            attn_flash = attn_flash,
            attn_qk_norm = attn_qk_norm,
            num_register_tokens = num_register_tokens,
            adaptive_rmsnorm = True,
            adaptive_rmsnorm_cond_dim_in = time_hidden_dim,
            use_gateloop_layers = use_gateloop_layers
        )

        dim_out = dim_in # expression code
        self.to_pred = nn.Linear(dim, dim_out, bias = False)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        # classifier-free gudiance
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1.:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x, # noised y0 of landmark
        *,
        times, # random in 0~1
        cond_audio, # driving audio
        self_attn_mask = None, # x_mask, since the length of samples in a batch are different
        cond_drop_prob = 0.1,
        target = None, # GT landmark, if None, infer mode
        cond = None, # reference landmark
        cond_mask = None, # mask that denotes frames to be predict as True
        ret=None
    ):
        if ret is None:
            ret = {}
        # project in, in case codebook dim is not equal to model dimensions
        # x 和 cond 是相同shape的，不同的是，x是target加噪声的结果，而cond是对target做mask后得到的reference。
        x = self.proj_in(x) 

        if exists(cond):
            cond = self.proj_in(cond)

        cond = default(cond, x) # x和cond的区别，见上面的分析

        # shapes
        batch, seq_len, cond_dim = cond.shape
        assert cond_dim == x.shape[-1]

        # auto manage shape of times, for odeint times

        if times.ndim == 0:
            times = repeat(times, '-> b', b = cond.shape[0])

        if times.ndim == 1 and times.shape[0] == 1:
            times = repeat(times, '1 -> b', b = cond.shape[0])

        # construct conditioning mask if not given
        if self.training:
            # 被mask住的就是要predict的部分
            if not exists(cond_mask):
                if coin_flip(): # 0.5 概率
                    frac_lengths = torch.zeros((batch,), device = self.device).float().uniform_(*self.frac_lengths_mask) # 0.7,1.0
                    # 这样得到的mask是连续的一个fraction
                    cond_mask = mask_from_frac_lengths(seq_len, frac_lengths)
                else:
                    # 这样得到的mask是散成豆花的
                    p_drop_prob_ = self.frac_lengths_mask[0] + random.random()*(self.frac_lengths_mask[1]-self.frac_lengths_mask[0])
                    cond_mask = prob_mask_like((batch, seq_len), p_drop_prob_, self.device)
                    # cond_mask = prob_mask_like((batch, seq_len), self.p_drop_prob, self.device)
        else:
            if not exists(cond_mask):
                # cond就是sample
                # 没有cond mask,代表没有reference audio, 所以直接mask住所有
                cond_mask = torch.ones((batch, seq_len), device = cond.device, dtype = torch.bool) 
        cond_mask_with_pad_dim = rearrange(cond_mask, '... -> ... 1') # 这个mask的意思是，True代表需要predict，False代表是reference

        # as described in section 3.2
        x = x * cond_mask_with_pad_dim # 这个是y0，源于noise，需要预测的部分保留为noise，不需要预测的reference部分被变成0
        cond = cond * ~cond_mask_with_pad_dim # 这个是reference音频, 所以标志出来需要pred的部分都变成0了

        # used by forward_with_cond_scale to achieve classifier free guidance
        # cond_drop_prob==1.0 denotes unconditional result
        if cond_drop_prob > 0.:
            cond_drop_mask = prob_mask_like(cond.shape[:1], cond_drop_prob, self.device) # 这个mask是散成豆花的

            # 随机对reference landmark 做dropout
            cond = torch.where(
                rearrange(cond_drop_mask, '... -> ... 1 1'), # cond
                self.null_cond, # fill true
                cond # fill false
            )

        # phoneme or semantic conditioning embedding
        cond_audio_emb = self.to_cond_emb(cond_audio)
        cond_audio_emb_length = cond_audio_emb.shape[-2]
        if cond_audio_emb_length != seq_len:
            cond_audio_emb = rearrange(cond_audio_emb, 'b n d -> b d n')
            cond_audio_emb = interpolate_1d(cond_audio_emb, seq_len)
            cond_audio_emb = rearrange(cond_audio_emb, 'b d n -> b n d')
            if exists(self_attn_mask):
                self_attn_mask = interpolate_1d(self_attn_mask, seq_len)

        # concat source signal, driving audio, and reference landmark
        # and project
        to_concat = [*filter(exists, (x, cond_audio_emb, cond))]
        embed = torch.cat(to_concat, dim = -1)

        x = self.to_embed(embed)

        x = self.conv_embed(x) + x

        time_emb = self.sinu_pos_emb(times)

        # attend

        x = self.transformer(
            x,
            mask = self_attn_mask,
            adaptive_rmsnorm_cond = time_emb
        )

        x = self.to_pred(x)
        # if no target passed in, just return logits
        ret['pred'] = x

        if not exists(target):
            # 不提供target，默认是infer模式，直接输出sample
            return x
        else:
            # 提供target，默认training模式，输出loss
            loss_mask = reduce_masks_with_and(cond_mask, self_attn_mask)
            if not exists(loss_mask):
                return F.mse_loss(x, target)
            ret['loss_mask'] = loss_mask
            loss = F.mse_loss(x, target, reduction = 'none')

            loss = reduce(loss, 'b n d -> b n', 'mean')
            loss = loss.masked_fill(~loss_mask, 0.)

            # masked mean

            num = reduce(loss, 'b n -> b', 'sum')
            den = loss_mask.sum(dim = -1).clamp(min = 1e-5)
            loss = num / den
            loss = loss.mean()
            ret['mse'] = loss
            return loss


if __name__ == '__main__':
    # Create an instance of the VoiceBox model
    model = InContextTransformerAudio2Motion()

    # Generate a random input tensor using torch.randn
    input_tensor = torch.randn(2, 125, 64)  # Assuming input shape is (batch_size, dim_in)
    time_tensor = torch.rand(2)  # Assuming input shape is (batch_size, dim_in)
    audio_tensor = torch.rand(2, 125, 1024)  # Assuming input shape is (batch_size, dim_in)

    # Pass the input tensor through the VoiceBox model
    output = model.forward_with_cond_scale(input_tensor, times=time_tensor, cond_audio=audio_tensor, cond=input_tensor)

    # Print the shape of the output tensor
    print(output.shape)


