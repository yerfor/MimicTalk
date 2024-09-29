import torch
import torch.nn as nn
import numpy as np
import random
from modules.audio2motion.cfm.icl_transformer import InContextTransformerAudio2Motion
from modules.audio2motion.cfm.cfm_wrapper import ConditionalFlowMatcherWrapper
from utils.commons.pitch_utils import f0_to_coarse


class InContextAudio2MotionModel(nn.Module):
    def __init__(self, mode='icl_transformer', hparams=None):
        super().__init__()    
        self.hparams = hparams
        feat_dim = 256
        self.hubert_encoder = nn.Sequential(*[
                nn.Conv1d(1024, feat_dim , 3, 1, 1, bias=False),
                nn.BatchNorm1d(feat_dim),
                nn.GELU(),
                nn.Conv1d(feat_dim, feat_dim, 3, 1, 1, bias=False)
        ])
        dim_audio_in = feat_dim
        if hparams.get("use_aux_features", False):
            aux_feat_dim = 32
            self.pitch_embed = nn.Embedding(300, aux_feat_dim, None)
            self.pitch_encoder = nn.Sequential(*[
                    nn.Conv1d(aux_feat_dim, aux_feat_dim , 3, 1, 1, bias=False),
                    nn.BatchNorm1d(aux_feat_dim),
                    nn.GELU(),
                    nn.Conv1d(aux_feat_dim, aux_feat_dim, 3, 1, 1, bias=False)
                ])

            self.blink_embed = nn.Embedding(2, aux_feat_dim)
            self.null_blink_embed = nn.Parameter(torch.randn(aux_feat_dim))

            self.mouth_amp_embed = nn.Parameter(torch.randn(aux_feat_dim))
            self.null_mouth_amp_embed = nn.Parameter(torch.randn(aux_feat_dim))
            dim_audio_in += 3 * aux_feat_dim

        icl_transformer = InContextTransformerAudio2Motion(
                                    dim_in=64 + 3 + 3, # exp and euler and trans
                                    dim_audio_in=dim_audio_in,
                                    dim=feat_dim,
                                    depth=16,
                                    dim_head=64,
                                    heads=8,
                                    frac_lengths_mask=(0.6, 1.),
                                    )
        self.mode = mode
        if mode == 'icl_transformer':
            self.backbone = icl_transformer
        elif mode == 'icl_flow_matching':
            flow_matching_model = ConditionalFlowMatcherWrapper(icl_transformer)
            self.backbone = flow_matching_model
        else:
            raise NotImplementedError()

        # used during inference
        self.hubert_context = None
        self.f0_context = None
        self.motion_context = None

    def num_params(self, model=None, print_out=True, model_name="model"):
        if model is None:
            model = self
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print(f'| {model_name} Trainable Parameters: %.3fM' % parameters)
        return parameters
    
    def device(self):
        return self.model.parameters().__next__().device

    def forward(self, batch, ret, train=True, temperature=1., cond_scale=1.0, denoising_steps=10):
        infer = not train
        hparams = self.hparams
        mask = batch['y_mask'].bool()
        mel = batch['audio']
        # f0 = batch['f0'] # [b,t]
        if 'blink' not in batch:
            batch['blink'] = torch.zeros([mel.shape[0], mel.shape[1]], dtype=torch.long, device=mel.device)
        blink = batch['blink']

        if 'mouth_amp' not in batch:
            batch['mouth_amp'] = torch.ones([mel.shape[0], 1], device=mel.device)
        mouth_amp = batch['mouth_amp']
            
        cond_mask = None
        cond = None
        if infer and self.hubert_context is not None:
            mel = torch.cat([self.hubert_context.to(mel.device), mel], dim=1)
            # f0 = torch.cat([self.f0_context.to(mel.device), f0], dim=1)
            blink = torch.cat([torch.zeros([mel.shape[0], self.hubert_context.shape[1],1],dtype=mel.dtype, device=mel.device), blink], dim=1)
            mask = torch.ones([mel.shape[0], mel.shape[1]//2,], dtype=mel.dtype, device=mel.device).bool()
            cond = torch.randn([mel.shape[0], mel.shape[1]//2, 64 + 6], dtype=mel.dtype, device=mel.device)
            if hparams.get("zero_input_for_transformer", True) and self.mode == 'icl_transformer':
                cond = cond * 0
            cond[:, :self.motion_context.shape[1]] = self.motion_context.to(mel.device)
            cond_mask = torch.ones([mel.shape[0], mel.shape[1]//2,], dtype=mel.dtype, device=mel.device) # 这个mask，1代表需要预测，0代表是reference
            cond_mask[:, :self.motion_context.shape[1]] = 0. # 将reference部分设置为0
            cond_mask = cond_mask.bool()
            
        cond_feat = self.hubert_encoder(mel.transpose(1,2)).transpose(1,2)
        cond_feats = [cond_feat]

        if hparams.get("use_aux_features", False):
            # use blink, f0, mouth_amp as auxiliary features in addtion to the hubert feature
            if (self.training and random.random() < 0.5) or (batch.get("null_cond", False)):
                use_null_aux_feats = True
            else:
                use_null_aux_feats = False

            # f0_coarse = f0_to_coarse(f0)
            # pitch_emb = self.pitch_embed(f0_coarse)
            # pitch_feat = self.pitch_encoder(pitch_emb.transpose(1,2)).transpose(1,2)

            if use_null_aux_feats:  
                mouth_amp_feat = self.null_mouth_amp_embed.unsqueeze(0).repeat([mel.shape[0], cond_feat.shape[1],1])
                blink_feat = self.null_blink_embed.unsqueeze(0).repeat([mel.shape[0], cond_feat.shape[1],1])
            else: 
                blink_feat = self.blink_embed(blink.squeeze(2).long())
                mouth_amp_feat = mouth_amp.unsqueeze(1) * self.mouth_amp_embed.unsqueeze(0)
                mouth_amp_feat = mouth_amp_feat.repeat([1,cond_feat.shape[1],1])

            cond_feats.append(pitch_feat)
            cond_feats.append(blink_feat)
            cond_feats.append(mouth_amp_feat)

        cond_feat = torch.cat(cond_feats, dim=-1)

        if not infer:
            # Train
            exp = batch['y']
            if self.mode == 'icl_transformer':
                x = torch.randn_like(exp)
                times_tensor = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
                if hparams.get("zero_input_for_transformer", True):
                    mse_loss = self.backbone(x=x*0, times=times_tensor, cond_audio=cond_feat, self_attn_mask=mask, cond_drop_prob=0., target=exp, cond=exp, cond_mask=None, ret=ret)
                else:
                    mse_loss = self.backbone(x=x, times=times_tensor, cond_audio=cond_feat, self_attn_mask=mask, cond_drop_prob=0., target=exp, cond=exp, cond_mask=None, ret=ret)
            elif self.mode == 'icl_flow_matching':
                mse_loss = self.backbone(x1=exp, cond_audio=cond_feat, mask=mask, cond=exp, cond_mask=None, ret=ret)
            
            ret['pred'] = ret['pred']
            ret['loss_mask'] = ret['loss_mask']
            ret['mse'] = mse_loss
            return mse_loss
        else:
            # Infer
            # todo: 在infer的时候能够使用上context，即提供cond_mask
            if cond is None:
                target_x_len = mask.shape[1]
                cond = torch.randn([cond_feat.shape[0], target_x_len, 64 + 6], dtype=cond_feat.dtype, device=cond_feat.device)
                if hparams.get("zero_input_for_transformer", True):
                    cond = cond * 0
            if self.mode == 'icl_transformer':
                times_tensor = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
                x_recon = self.backbone(x=cond, times=times_tensor, cond_audio=cond_feat, self_attn_mask=mask, cond_drop_prob=0., cond=cond, cond_mask=cond_mask, ret=ret)
            elif self.mode == 'icl_flow_matching':
                # default of voicebox is steps=3, elapsed time 0.56s; as for our steps=1000, elapsed time 0.66s
                x_recon = self.backbone.sample(cond_audio=cond_feat, self_attn_mask=mask, cond=cond, cond_mask=cond_mask, temperature=temperature, steps=denoising_steps, cond_scale=cond_scale)
                # x_recon = self.backbone.sample(cond_audio=cond_feat, self_attn_mask=mask, cond=cond, cond_mask=cond_mask, temperature=temperature, steps=5, )
            x_recon = x_recon * mask.unsqueeze(-1)
                
            ret['pred'] = x_recon
            ret['mask'] = mask
            
            if self.motion_context is not None:
                len_reference = self.motion_context.shape[1]
                ret['pred'] = x_recon[:,len_reference:]
                ret['mask'] = mask[:,len_reference:]

            return x_recon

    def add_sample_to_context(self, motion, hubert=None, f0=None):
        # B, T, C, audio should 2X length of motion
        assert motion is not None

        if self.motion_context is None:
            self.motion_context = motion
        else:
            self.motion_context = torch.cat([self.motion_context, motion], dim=1)
        if self.hubert_context is None:
            if hubert is None:
                self.hubert_context = torch.zeros([motion.shape[0], motion.shape[1]*2, 1024], dtype=motion.dtype, device=motion.device)
                self.f0_context = torch.zeros([motion.shape[0], motion.shape[1]*2], dtype=motion.dtype, device=motion.device)
            else:
                self.hubert_context = hubert
                self.f0_context = f0.reshape([hubert.shape[0],hubert.shape[1]])
        else:
            if hubert is None:
                self.hubert_context = torch.cat([self.hubert_context, torch.zeros([motion.shape[0], motion.shape[1]*2, 1024], dtype=motion.dtype, device=motion.device)], dim=1)
                self.f0_context = torch.cat([self.f0_context, torch.zeros([motion.shape[0], motion.shape[1]*2], dtype=motion.dtype, device=motion.device)], dim=1)
            else:      
                self.hubert_context = torch.cat([self.hubert_context, hubert], dim=1)
                self.f0_context = torch.cat([self.f0_context, f0], dim=1)
        return 0

    def empty_context(self):
        self.hubert_context = None
        self.f0_context = None
        self.motion_context = None
# 
if __name__ == '__main__':
    model = InContextAudio2MotionModel()
    model.num_params()