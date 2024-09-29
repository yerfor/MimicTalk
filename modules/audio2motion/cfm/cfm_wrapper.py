import math
import time
import torch
from torch import nn, Tensor, einsum, IntTensor, FloatTensor, BoolTensor
from torch.nn import Module
import torch.nn.functional as F
import torchode
from torchdiffeq import odeint

from beartype import beartype
from beartype.typing import Tuple, Optional, List, Union

from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, pack, unpack

from modules.audio2motion.cfm.utils import *
from modules.audio2motion.cfm.icl_transformer import InContextTransformerAudio2Motion


# wrapper for the CNF
def is_probably_audio_from_shape(t):
    return exists(t) and (t.ndim == 2 or (t.ndim == 3 and t.shape[1] == 1))


class ConditionalFlowMatcherWrapper(Module):
    @beartype
    def __init__(
        self,
        icl_transformer_model: InContextTransformerAudio2Motion = None,
        sigma = 0.,
        ode_atol = 1e-5,
        ode_rtol = 1e-5,
        # ode_step_size = 0.0625,
        use_torchode = False,
        torchdiffeq_ode_method = 'midpoint',   # use midpoint for torchdiffeq, as in paper
        torchode_method_klass = torchode.Tsit5,      # use tsit5 for torchode, as torchode does not have midpoint (recommended by Bryan @b-chiang)
        cond_drop_prob = 0.
    ):
        super().__init__()
        self.sigma = sigma
        if icl_transformer_model is None:
            icl_transformer_model = InContextTransformerAudio2Motion()
        self.icl_transformer_model = icl_transformer_model
        self.cond_drop_prob = cond_drop_prob
        self.use_torchode = use_torchode
        self.torchode_method_klass = torchode_method_klass
        self.odeint_kwargs = dict(
            atol = ode_atol,
            rtol = ode_rtol,
            method = torchdiffeq_ode_method,
            # options = dict(step_size = ode_step_size)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def sample(
        self,
        *,
        cond_audio = None, # [B, T (可以是2倍，会被interpolate到x1的length), C]
        cond = None, # random
        cond_mask = None,
        steps = 3, # flow steps, 3和10都需要0.56s
        cond_scale = 1.,
        ret=None,
        self_attn_mask = None,
        temperature=1.0,
    ):
        if ret is None:
            ret = {}
        cond_target_length = cond_audio.shape[1] // 2
        if exists(cond):
            cond = curtail_or_pad(cond, cond_target_length)
        else:
            cond = torch.zeros((cond_audio.shape[0], cond_target_length, self.dim_cond_emb), device = self.device)

        shape = cond.shape
        batch = shape[0]

        # neural ode

        self.icl_transformer_model.eval()

        def fn(t, x, *, packed_shape = None):
            if exists(packed_shape):
                x = unpack_one(x, packed_shape, 'b *')

            out = self.icl_transformer_model.forward_with_cond_scale(
                x, # rand
                times = t, # timestep in DM
                cond_audio = cond_audio,
                cond = cond, # rand?
                cond_scale = cond_scale,
                cond_mask = cond_mask,
                self_attn_mask = self_attn_mask,
                ret=ret,
            )

            if exists(packed_shape):
                out = rearrange(out, 'b ... -> b (...)')

            return out

        y0 = torch.randn_like(cond) * float(temperature)
        t = torch.linspace(0, 1, steps, device = self.device)
        timestamp_before_sampling = time.time()
        if not self.use_torchode:
            print(f'sampling based on torchdiffeq with flow total_steps={steps}')

            trajectory = odeint(fn, y0, t, **self.odeint_kwargs) # 从y0位置出发，fn根据当前位置提供velocity，沿着t进行积分。
            sampled = trajectory[-1]
        else:
            print(f'sampling based on torchode with flow total_steps={steps}')

            t = repeat(t, 'n -> b n', b = batch)
            y0, packed_shape = pack_one(y0, 'b *')

            fn = partial(fn, packed_shape = packed_shape)

            term = to.ODETerm(fn)
            step_method = self.torchode_method_klass(term = term)

            step_size_controller = to.IntegralController(
                atol = self.odeint_kwargs['atol'],
                rtol = self.odeint_kwargs['rtol'],
                term = term
            )

            solver = to.AutoDiffAdjoint(step_method, step_size_controller)
            jit_solver = torch.compile(solver)

            init_value = to.InitialValueProblem(y0 = y0, t_eval = t)

            sol = jit_solver.solve(init_value)

            sampled = sol.ys[:, -1]
            sampled = unpack_one(sampled, packed_shape, 'b *')

        print(f"Flow matching sampling process elapsed in {time.time()-timestamp_before_sampling:.4f} second")
        return sampled


    def forward(
        self,
        x1, # gt sample, landmark, [B, T, C]
        *,
        mask = None, # mask of frames in batch
        cond_audio = None, # [B, T (可以是2倍，会被interpolate到x1的length), C]
        cond = None, # reference landmark
        cond_mask = None, # mask of reference landmark, reference are marked as False, and frames to be predicted are True
        ret = None,
    ):
        """
        training step of Continous Normalizing Flow
        following eq (5) (6) in https://arxiv.org/pdf/2306.15687.pdf
        """
        if ret is None:
            ret = {}
        batch, seq_len, dtype, sigma_ = *x1.shape[:2], x1.dtype, self.sigma

        # main conditional flow logic is below
        # x0 is gaussian noise
        x0 = torch.randn_like(x1)
        # batch-wise random times with 0~1
        times = torch.rand((batch,), dtype = dtype, device = self.device)
        t = rearrange(times, 'b -> b 1 1')

        # sample xt within x0=>xt=>x1 (Sec 3.1 in the paper)
        # The associated conditional vector field is ut(x | x1) = (x1 − (1 − σmin)*x) / (1 − (1 − σmin)*t), 
        # and the conditional flow is φt(x | x1) = (1 − (1 − σmin)*t)*x + t * x1.
        current_position_in_flows = (1 - (1 - sigma_) * t) * x0 + t * x1 # input of the transformer, noised sample, conditional flow, φt(x | x1) in FlowMatching
        optimal_path = x1 - (1 - sigma_) * x0 # target of the transformer, vector field , u_t(x|x1) in FlowMatching

        # predict
        self.icl_transformer_model.train()
        # the ouput of transformer is learnable vector field v_t(x;theta) in FlowMatching
        loss = self.icl_transformer_model(
            current_position_in_flows, # noised motion sample
            cond = cond,
            cond_mask = cond_mask, 
            times = times,
            target = optimal_path, # 
            self_attn_mask = mask,
            cond_audio = cond_audio,
            cond_drop_prob = self.cond_drop_prob,
            ret=ret,
        )

        pred_x1_minus_x0 = ret['pred'] # predicted path
        pred_x1 = pred_x1_minus_x0 + (1 - sigma_) * x0
        ret['pred'] = pred_x1
        return loss

if __name__ == '__main__':
    icl_transformer = InContextTransformerAudio2Motion()
    model = ConditionalFlowMatcherWrapper(icl_transformer)
    x = torch.randn([2, 125, 64])
    cond = torch.randn([2, 125, 64])
    cond_audio = torch.randn([2, 250, 1024])
    y = model(x, cond=cond, cond_audio=cond_audio)
    y = model.sample(cond=cond, cond_audio=cond_audio)
    print(y.shape)