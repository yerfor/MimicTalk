"""
将One-shot的说话人大模型(os_secc2plane or os_secc2plane_torso)在单一说话人(一张照片或一段视频)上overfit, 实现和GeneFace++类似的效果
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import random
import time
import numpy as np
import importlib
import tqdm
import copy
import cv2
import glob
import imageio
# common utils
from utils.commons.hparams import hparams, set_hparams
from utils.commons.tensor_utils import move_to_cuda, convert_to_tensor
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint
# 3DMM-related utils
from deep_3drecon.deep_3drecon_models.bfm import ParametricFaceModel
from data_util.face3d_helper import Face3DHelper
from data_gen.utils.process_image.fit_3dmm_landmark import fit_3dmm_for_a_image
from data_gen.utils.process_video.fit_3dmm_landmark import fit_3dmm_for_a_video
from data_gen.utils.process_video.extract_segment_imgs import decode_segmap_mask_from_image
from deep_3drecon.secc_renderer import SECC_Renderer
from data_gen.eg3d.convert_to_eg3d_convention import get_eg3d_convention_camera_pose_intrinsic
from data_gen.runs.binarizer_nerf import get_lip_rect
# Face Parsing 
from data_gen.utils.mp_feature_extractors.mp_segmenter import MediapipeSegmenter
from data_gen.utils.process_video.extract_segment_imgs import inpaint_torso_job, extract_background
# other inference utils
from inference.infer_utils import mirror_index, load_img_to_512_hwc_array, load_img_to_normalized_512_bchw_tensor
from inference.infer_utils import smooth_camera_sequence, smooth_features_xd
from inference.edit_secc import blink_eye_for_secc, hold_eye_opened_for_secc
from modules.commons.loralib.utils import mark_only_lora_as_trainable
from utils.nn.model_utils import num_params
import lpips
from utils.commons.meters import AvgrageMeter
meter = AvgrageMeter()
from torch.utils.tensorboard import SummaryWriter
class LoRATrainer(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.inp = inp
        self.lora_args = {'lora_mode': inp['lora_mode'], 'lora_r': inp['lora_r']}
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        head_model_dir = inp['head_ckpt']
        torso_model_dir = inp['torso_ckpt']
        model_dir = torso_model_dir if torso_model_dir != '' else head_model_dir
        cmd = f"cp {os.path.join(model_dir, 'config.yaml')} {self.inp['work_dir']}"
        print(cmd)
        os.system(cmd)
        with open(os.path.join(self.inp['work_dir'], 'config.yaml'), "a") as f:
            f.write(f"\nlora_r: {inp['lora_r']}")
            f.write(f"\nlora_mode: {inp['lora_mode']}")
            f.write(f"\n")
        self.secc2video_model = self.load_secc2video(model_dir)
        self.secc2video_model.to(device).eval()
        self.seg_model = MediapipeSegmenter()
        self.secc_renderer = SECC_Renderer(512)
        self.face3d_helper = Face3DHelper(use_gpu=True, keypoint_mode='lm68')
        self.mp_face3d_helper = Face3DHelper(use_gpu=True, keypoint_mode='mediapipe')
        # self.camera_selector = KNearestCameraSelector()
        self.load_training_data(inp)
    def load_secc2video(self, model_dir):
        inp = self.inp
        from modules.real3d.secc_img2plane_torso import OSAvatarSECC_Img2plane, OSAvatarSECC_Img2plane_Torso
        hp = set_hparams(f"{model_dir}/config.yaml", print_hparams=False, global_hparams=True)
        hp['htbsr_head_threshold'] = 1.0
        if 'torso' in hp['task_cls'].lower():
            self.torso_mode = True
            model = OSAvatarSECC_Img2plane_Torso(hp=hp, lora_args=self.lora_args)
        else:
            self.torso_mode = False
            model = OSAvatarSECC_Img2plane(hp=hp, lora_args=self.lora_args)
        mark_only_lora_as_trainable(model, bias='none')
        lora_ckpt_path = os.path.join(inp['work_dir'], 'checkpoint.ckpt')
        if os.path.exists(lora_ckpt_path):
            self.learnable_triplane = nn.Parameter(torch.zeros([1, 3, model.triplane_hid_dim*model.triplane_depth, 256, 256]).float().cuda(), requires_grad=True)
            model._last_cano_planes = self.learnable_triplane
            load_ckpt(model, lora_ckpt_path, model_name='model', strict=False)   
        else:
            load_ckpt(model, f"{model_dir}", model_name='model', strict=False)   
            
        num_params(model)
        self.model = model 
        return model
    def load_training_data(self, inp):
        video_id = inp['video_id']
        if video_id.endswith((".mp4", ".png", ".jpg", ".jpeg")):
            # If input video is not GeneFace training videos, convert it into GeneFace convention
            video_id_ = video_id
            video_id = os.path.basename(video_id)[:-4]
            inp['video_id'] = video_id
            target_video_path = f'data/raw/videos/{video_id}.mp4'
            if not os.path.exists(target_video_path):
                print(f"| Copying video to {target_video_path}")
                os.makedirs(os.path.dirname(target_video_path), exist_ok=True)
                cmd = f"ffmpeg -i {video_id_} -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 -y {target_video_path}"
                print(f"| {cmd}")
                os.system(cmd)
        target_video_path = f'data/raw/videos/{video_id}.mp4'
        print(f"| Copy source video into work dir: {self.inp['work_dir']}")
        os.system(f"cp {target_video_path} {self.inp['work_dir']}")
        # check head_img path
        head_img_pattern = f'data/processed/videos/{video_id}/head_imgs/*.png'
        head_img_names = sorted(glob.glob(head_img_pattern))
        if len(head_img_names) == 0:
            # extract head_imgs
            head_img_dir = os.path.dirname(head_img_pattern)
            print(f"| Pre-extracted head_imgs not found, try to extract and save to {head_img_dir}, this may take a while...")
            gt_img_dir = f"data/processed/videos/{video_id}/gt_imgs"
            os.makedirs(gt_img_dir, exist_ok=True)
            target_video_path = f'data/raw/videos/{video_id}.mp4'
            cmd = f"ffmpeg -i {target_video_path} -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 -start_number 0 -y {gt_img_dir}/%08d.jpg"
            print(f"| {cmd}")
            os.system(cmd)
            # extract image, segmap, and background
            cmd = f"python data_gen/utils/process_video/extract_segment_imgs.py --ds_name=nerf --vid_dir={target_video_path}" 
            print(f"| {cmd}")
            os.system(cmd)
            print("| Head images Extracted!")
        num_samples = len(head_img_names)
        npy_name = f"data/processed/videos/{video_id}/coeff_fit_mp_for_lora.npy"
        if os.path.exists(npy_name):
            coeff_dict = np.load(npy_name, allow_pickle=True).tolist()
        else:
            print(f"| Pre-extracted 3DMM coefficient not found, try to extract and save to {npy_name}, this may take a while...")
            coeff_dict = fit_3dmm_for_a_video(f'data/raw/videos/{video_id}.mp4', save=False)
            os.makedirs(os.path.dirname(npy_name), exist_ok=True)
            np.save(npy_name, coeff_dict)
        ids = convert_to_tensor(coeff_dict['id']).reshape([-1,80]).cuda()
        exps = convert_to_tensor(coeff_dict['exp']).reshape([-1,64]).cuda()
        eulers = convert_to_tensor(coeff_dict['euler']).reshape([-1,3]).cuda()
        trans = convert_to_tensor(coeff_dict['trans']).reshape([-1,3]).cuda()
        WH = 512 # now we only support 512x512
        lm2ds = WH * self.face3d_helper.reconstruct_lm2d(ids, exps, eulers, trans).cpu().numpy()
        lip_rects = [get_lip_rect(lm2ds[i], WH, WH) for i in range(len(lm2ds))]
        kps = self.face3d_helper.reconstruct_lm2d(ids, exps, eulers, trans).cuda()
        kps = (kps-0.5) / 0.5 # rescale to -1~1
        kps = torch.cat([kps, torch.zeros([*kps.shape[:-1], 1]).cuda()], dim=-1)
        camera_ret = get_eg3d_convention_camera_pose_intrinsic({'euler': torch.tensor(coeff_dict['euler']).reshape([-1,3]), 'trans': torch.tensor(coeff_dict['trans']).reshape([-1,3])})
        c2w, intrinsics = camera_ret['c2w'], camera_ret['intrinsics']
        cameras = torch.tensor(np.concatenate([c2w.reshape([-1,16]), intrinsics.reshape([-1,9])], axis=-1)).cuda()
        camera_smo_ksize = 7
        cameras = smooth_camera_sequence(cameras.cpu().numpy(), kernel_size=camera_smo_ksize) # [T, 25]
        cameras = torch.tensor(cameras).cuda()
        zero_eulers = eulers * 0
        zero_trans = trans * 0
        _, cano_secc_color = self.secc_renderer(ids[0:1], exps[0:1]*0, zero_eulers[0:1], zero_trans[0:1])
        src_idx = 0
        _, src_secc_color = self.secc_renderer(ids[0:1], exps[src_idx:src_idx+1], zero_eulers[0:1], zero_trans[0:1])
        drv_secc_colors = [None for _ in range(len(exps))]
        drv_head_imgs = [None for _ in range(len(exps))]
        drv_torso_imgs = [None for _ in range(len(exps))]
        drv_com_imgs = [None for _ in range(len(exps))]
        segmaps = [None for _ in range(len(exps))]
        img_name = f'data/processed/videos/{video_id}/bg.jpg'
        bg_img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
        ds = {
            'id': ids.cuda().float(),
            'exps': exps.cuda().float(),
            'eulers': eulers.cuda().float(),
            'trans': trans.cuda().float(),
            'cano_secc_color': cano_secc_color.cuda().float(),
            'src_secc_color': src_secc_color.cuda().float(),
            'cameras': cameras.float(),
            'video_id': video_id,
            'lip_rects': lip_rects,
            'head_imgs': drv_head_imgs,
            'torso_imgs': drv_torso_imgs,
            'com_imgs': drv_com_imgs,
            'bg_img': bg_img,
            'segmaps': segmaps,
            'kps': kps,
        }
        self.ds = ds
        return ds
    def training_loop(self, inp):
        video_id = self.ds['video_id']
        lora_params = [p for k, p in self.secc2video_model.named_parameters() if 'lora_' in k]
        self.criterion_lpips = lpips.LPIPS(net='alex',lpips=True).cuda()
        self.logger = SummaryWriter(log_dir=inp['work_dir'])
        if not hasattr(self, 'learnable_triplane'):
            src_idx = 0 # init triplane from the first frame's prediction
            self.learnable_triplane = nn.Parameter(torch.zeros([1, 3, self.secc2video_model.triplane_hid_dim*self.secc2video_model.triplane_depth, 256, 256]).float().cuda(), requires_grad=True)
            img_name = f'data/processed/videos/{video_id}/head_imgs/{format(src_idx, "08d")}.png'
            img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float().cuda().float() # [3, H, W]
            cano_plane = self.secc2video_model.cal_cano_plane(img.unsqueeze(0)) # [1, 3, CD, h, w]
            self.learnable_triplane.data = cano_plane.data
            self.secc2video_model._last_cano_planes = self.learnable_triplane
        if len(lora_params) == 0:
            self.optimizer = torch.optim.AdamW([self.learnable_triplane], lr=inp['lr_triplane'], weight_decay=0.01, betas=(0.9,0.98))
        else:
            self.optimizer = torch.optim.Adam(lora_params, lr=inp['lr'], betas=(0.9,0.98))
            self.optimizer.add_param_group({
                'params': [self.learnable_triplane],
                'lr': inp['lr_triplane'],
                'betas': (0.9, 0.98)
            })
        
        ids = self.ds['id']
        exps = self.ds['exps']
        zero_eulers = self.ds['eulers']*0
        zero_trans = self.ds['trans']*0
        num_updates = inp['max_updates']
        batch_size = inp['batch_size'] # 1 for lower gpu mem usage
        num_samples = len(self.ds['cameras'])
        init_plane = self.learnable_triplane.detach().clone()
        if num_samples <= 5:
            lambda_reg_triplane = 1.0
        elif num_samples <= 250:
            lambda_reg_triplane = 0.1
        else:
            lambda_reg_triplane = 0.
        for i_step in tqdm.trange(num_updates+1,desc="training lora..."):
            milestone_steps = [100, 200, 500]
            if i_step % 1000 == 0 or i_step in milestone_steps:
                trainer.test_loop(inp, step=i_step)
                if i_step != 0:
                    filepath = os.path.join(inp['work_dir'], f"model_ckpt_steps_{i_step}.ckpt") 
                    checkpoint = self.dump_checkpoint(inp)
                    tmp_path = str(filepath) + ".part"
                    torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=False)
                    os.replace(tmp_path, filepath)
                
            drv_idx = [random.randint(0, num_samples-1) for _ in range(batch_size)]
            drv_secc_colors = []
            gt_imgs = []
            head_imgs = []
            segmaps_0 = []
            segmaps = []
            torso_imgs = []
            drv_lip_rects = []
            kp_src = []
            kp_drv = []
            for di in drv_idx:
                # 读取target image
                if self.torso_mode:
                    if self.ds['com_imgs'][di] is None:
                        # img_name = f'data/processed/videos/{video_id}/gt_imgs/{format(di, "08d")}.jpg'
                        img_name = f'data/processed/videos/{video_id}/com_imgs/{format(di, "08d")}.jpg'
                        img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
                        self.ds['com_imgs'][di] = img
                    gt_imgs.append(self.ds['com_imgs'][di])
                else:
                    if self.ds['head_imgs'][di] is None:
                        img_name = f'data/processed/videos/{video_id}/head_imgs/{format(di, "08d")}.png'
                        img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
                        self.ds['head_imgs'][di] = img
                    gt_imgs.append(self.ds['head_imgs'][di])
                if self.ds['head_imgs'][di] is None:
                    img_name = f'data/processed/videos/{video_id}/head_imgs/{format(di, "08d")}.png'
                    img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
                    self.ds['head_imgs'][di] = img
                head_imgs.append(self.ds['head_imgs'][di])
                # 使用第一帧的torso作为face v2v的输入
                if self.ds['torso_imgs'][0] is None:
                    img_name = f'data/processed/videos/{video_id}/inpaint_torso_imgs/{format(0, "08d")}.png'
                    img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
                    self.ds['torso_imgs'][0] = img
                torso_imgs.append(self.ds['torso_imgs'][0])
                # 所以segmap也用第一帧的了
                if self.ds['segmaps'][0] is None:
                    img_name = f'data/processed/videos/{video_id}/segmaps/{format(0, "08d")}.png'
                    seg_img = cv2.imread(img_name)[:,:, ::-1]
                    segmap = torch.from_numpy(decode_segmap_mask_from_image(seg_img)) # [6, H, W]
                    self.ds['segmaps'][0] = segmap
                segmaps_0.append(self.ds['segmaps'][0])
                if self.ds['segmaps'][di] is None:
                    img_name = f'data/processed/videos/{video_id}/segmaps/{format(di, "08d")}.png'
                    seg_img = cv2.imread(img_name)[:,:, ::-1]
                    segmap = torch.from_numpy(decode_segmap_mask_from_image(seg_img)) # [6, H, W]
                    self.ds['segmaps'][di] = segmap
                segmaps.append(self.ds['segmaps'][di])
                _, secc_color = self.secc_renderer(ids[0:1], exps[di:di+1], zero_eulers[0:1], zero_trans[0:1])
                drv_secc_colors.append(secc_color)
                drv_lip_rects.append(self.ds['lip_rects'][di])
                kp_src.append(self.ds['kps'][0])
                kp_drv.append(self.ds['kps'][di])
            bg_img = self.ds['bg_img'].unsqueeze(0).repeat([batch_size, 1, 1, 1]).cuda()
            ref_torso_imgs = torch.stack(torso_imgs).float().cuda()
            kp_src = torch.stack(kp_src).float().cuda()
            kp_drv = torch.stack(kp_drv).float().cuda()
            segmaps = torch.stack(segmaps).float().cuda()
            segmaps_0 = torch.stack(segmaps_0).float().cuda()
            tgt_imgs = torch.stack(gt_imgs).float().cuda()
            head_imgs = torch.stack(head_imgs).float().cuda()
            drv_secc_color = torch.cat(drv_secc_colors)
            cano_secc_color = self.ds['cano_secc_color'].repeat([batch_size, 1, 1, 1])
            src_secc_color = self.ds['src_secc_color'].repeat([batch_size, 1, 1, 1])
            cond = {'cond_cano': cano_secc_color,'cond_src': src_secc_color, 'cond_tgt': drv_secc_color,
                    'ref_torso_img': ref_torso_imgs, 'bg_img': bg_img, 
                    'segmap': segmaps_0, # v2v使用第一帧的torso作为source image来warp
                    'kp_s': kp_src, 'kp_d': kp_drv}
            camera = self.ds['cameras'][drv_idx]
            gen_output = self.secc2video_model.forward(img=None, camera=camera, cond=cond, ret={}, cache_backbone=False, use_cached_backbone=True)
            pred_imgs = gen_output['image']
            pred_imgs_raw = gen_output['image_raw']
            total_loss = 0
            occlusion_reg_l1 = gen_output.get("losses", {}).get('facev2v/occlusion_reg_l1', 0.)
            occlusion_2_reg_l1 = gen_output.get("losses", {}).get('facev2v/occlusion_2_reg_l1', 0.)
            occlusion_2_weights_entropy = gen_output.get("losses", {}).get('facev2v/occlusion_2_weights_entropy', 0.)
            total_loss += occlusion_reg_l1 * 0.001 + occlusion_2_reg_l1 * 0.001 + occlusion_2_weights_entropy * hparams['lam_occlusion_weights_entropy']
            # Weights Reg loss in torso
            # alphas = gen_output['weights_img'].clamp(1e-5, 1 - 1e-5)
            # loss_weights_entropy = torch.mean(- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas))
            # mv_head_masks = segmaps[:, [1,3,5]].sum(dim=1)
            # mv_head_masks_raw = F.interpolate(mv_head_masks.unsqueeze(1), size=(128,128)).squeeze(1)
            # face_mask = mv_head_masks_raw.bool().unsqueeze(1)
            # nonface_mask = ~ face_mask
            # loss_weights_l2_loss = (alphas[nonface_mask]-0).pow(2).mean() + (alphas[face_mask]-1).pow(2).mean()
            # total_loss = total_loss + loss_weights_entropy * 0.001 + loss_weights_l2_loss * 0.01
            # RGB reconstruction loss, 在raw上的额外loss会导致头部外有虚影
            # mse_loss = (pred_imgs - tgt_imgs).abs().mean()
            # lpips_loss = self.criterion_lpips(pred_imgs, tgt_imgs).mean() 
            mse_loss = (pred_imgs - tgt_imgs).abs().mean() + 0.2 * (pred_imgs_raw - F.interpolate(head_imgs, size=(128,128), mode='bilinear', antialias=True)).abs().mean()
            lpips_loss = self.criterion_lpips(pred_imgs, tgt_imgs).mean() + 0.2 * self.criterion_lpips(pred_imgs_raw, F.interpolate(head_imgs, size=(128,128), mode='bilinear', antialias=True)).mean()
            lip_mse_loss = 0
            lip_lpips_loss = 0
            for i in range(len(drv_idx)):
                xmin, xmax, ymin, ymax = drv_lip_rects[i]
                lip_tgt_imgs = tgt_imgs[i:i+1,:, ymin:ymax,xmin:xmax].contiguous()
                lip_pred_imgs = pred_imgs[i:i+1,:, ymin:ymax,xmin:xmax].contiguous()
                try:
                    lip_mse_loss = lip_mse_loss + (lip_pred_imgs - lip_tgt_imgs).abs().mean()
                    lip_lpips_loss = lip_lpips_loss + self.criterion_lpips(lip_pred_imgs, lip_tgt_imgs).mean()
                except: pass 
            total_loss = total_loss + mse_loss * 1.0 + lip_mse_loss * 0.2  + lpips_loss * inp['lambda_lpips'] + 0.2 * lip_lpips_loss * inp['lambda_lpips']
            # eye blink reg loss
            if i_step % 4 == 0:
                blink_secc_lst1 = []
                blink_secc_lst2 = []
                blink_secc_lst3 = []
                for i in range(len(drv_secc_color)):
                    secc = drv_secc_color[i]
                    blink_percent1 = random.random() * 0.5 # 0~0.5
                    blink_percent3 = 0.5 + random.random() * 0.5 # 0.5~1.0
                    blink_percent2 = (blink_percent1 + blink_percent3)/2
                    try:
                        out_secc1 = blink_eye_for_secc(secc, blink_percent1).to(secc.device)
                        out_secc2 = blink_eye_for_secc(secc, blink_percent2).to(secc.device)
                        out_secc3 = blink_eye_for_secc(secc, blink_percent3).to(secc.device)
                    except:
                        print("blink eye for secc failed, use original secc")
                        out_secc1 = copy.deepcopy(secc)
                        out_secc2 = copy.deepcopy(secc)
                        out_secc3 = copy.deepcopy(secc)
                    blink_secc_lst1.append(out_secc1)
                    blink_secc_lst2.append(out_secc2)
                    blink_secc_lst3.append(out_secc3)
                src_secc_color1 = torch.stack(blink_secc_lst1)
                src_secc_color2 = torch.stack(blink_secc_lst2)
                src_secc_color3 = torch.stack(blink_secc_lst3)
            blink_cond1 = {'cond_cano': cano_secc_color, 'cond_src': src_secc_color, 'cond_tgt': src_secc_color1}
            blink_cond2 = {'cond_cano': cano_secc_color, 'cond_src': src_secc_color, 'cond_tgt': src_secc_color2}
            blink_cond3 = {'cond_cano': cano_secc_color, 'cond_src': src_secc_color, 'cond_tgt': src_secc_color3}
            blink_secc_plane1 = self.model.cal_secc_plane(blink_cond1)
            blink_secc_plane2 = self.model.cal_secc_plane(blink_cond2)
            blink_secc_plane3 = self.model.cal_secc_plane(blink_cond3)
            interpolate_blink_secc_plane = (blink_secc_plane1 + blink_secc_plane3) / 2
            blink_reg_loss = torch.nn.functional.l1_loss(blink_secc_plane2, interpolate_blink_secc_plane)
            # lambda_reg_blink = 0.001
            lambda_reg_blink = 0.
            total_loss = total_loss + lambda_reg_blink
            total_loss = total_loss + lambda_reg_blink * blink_reg_loss

            # Triplane Reg loss
            triplane_reg_loss = (self.learnable_triplane - init_plane).abs().mean()
            total_loss = total_loss + triplane_reg_loss * lambda_reg_triplane
            # Update weights
            self.optimizer.zero_grad()
            total_loss.backward()
            self.learnable_triplane.grad.data = self.learnable_triplane.grad.data * self.learnable_triplane.numel()
            self.optimizer.step()
            meter.update(total_loss.item())
            if i_step % 100 == 0:
                print(f"Iter {i_step+1}: {meter.avg}")
                self.logger.add_scalar("loss", meter.avg, i_step)
                meter.reset()
    @torch.no_grad()
    def test_loop(self, inp, step=''):
        self.model.eval()
        # coeff_dict = np.load('data/processed/videos/Lieu/coeff_fit_mp_for_lora.npy', allow_pickle=True).tolist()
        # drv_exps = torch.tensor(coeff_dict['exp']).cuda().float()
        drv_exps = self.ds['exps']
        zero_eulers = self.ds['eulers']*0
        zero_trans = self.ds['trans']*0
        batch_size = 1
        num_samples = len(self.ds['cameras'])
        video_writer = imageio.get_writer(os.path.join(inp['work_dir'], f'val_step{step}.mp4'), fps=25)
        total_iters = min(num_samples, 250)
        video_id = inp['video_id']
        for i in tqdm.trange(total_iters,desc="testing lora..."):
            drv_idx = [i]
            drv_secc_colors = []
            gt_imgs = []
            segmaps = []
            torso_imgs = []
            drv_lip_rects = []
            kp_src = []
            kp_drv = []
            for di in drv_idx:
                # 读取target image
                if self.torso_mode:
                    if self.ds['com_imgs'][di] is None:
                        img_name = f'data/processed/videos/{video_id}/com_imgs/{format(di, "08d")}.jpg'
                        img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
                        self.ds['com_imgs'][di] = img
                    gt_imgs.append(self.ds['com_imgs'][di])
                else:
                    if self.ds['head_imgs'][di] is None:
                        img_name = f'data/processed/videos/{video_id}/head_imgs/{format(di, "08d")}.png'
                        img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
                        self.ds['head_imgs'][di] = img
                    gt_imgs.append(self.ds['head_imgs'][di])
                # 使用第一帧的torso作为face v2v的输入 
                if self.ds['torso_imgs'][0] is None:
                    img_name = f'data/processed/videos/{video_id}/inpaint_torso_imgs/{format(0, "08d")}.png'
                    img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
                    self.ds['torso_imgs'][0] = img
                torso_imgs.append(self.ds['torso_imgs'][0])
                # 所以segmap也用第一帧的了
                if self.ds['segmaps'][0] is None:
                    img_name = f'data/processed/videos/{video_id}/segmaps/{format(0, "08d")}.png'
                    seg_img = cv2.imread(img_name)[:,:, ::-1]
                    segmap = torch.from_numpy(decode_segmap_mask_from_image(seg_img)) # [6, H, W]
                    self.ds['segmaps'][0] = segmap
                segmaps.append(self.ds['segmaps'][0])
                drv_lip_rects.append(self.ds['lip_rects'][di])
                kp_src.append(self.ds['kps'][0])
                kp_drv.append(self.ds['kps'][di])
            bg_img = self.ds['bg_img'].unsqueeze(0).repeat([batch_size, 1, 1, 1]).cuda()
            ref_torso_imgs = torch.stack(torso_imgs).float().cuda()
            kp_src = torch.stack(kp_src).float().cuda()
            kp_drv = torch.stack(kp_drv).float().cuda()
            segmaps = torch.stack(segmaps).float().cuda()
            tgt_imgs = torch.stack(gt_imgs).float().cuda()
            for di in drv_idx:
                _, secc_color = self.secc_renderer(self.ds['id'][0:1], drv_exps[di:di+1], zero_eulers[0:1], zero_trans[0:1])
                drv_secc_colors.append(secc_color)
            drv_secc_color = torch.cat(drv_secc_colors)
            cano_secc_color = self.ds['cano_secc_color'].repeat([batch_size, 1, 1, 1])
            src_secc_color = self.ds['src_secc_color'].repeat([batch_size, 1, 1, 1])
            cond = {'cond_cano': cano_secc_color,'cond_src': src_secc_color, 'cond_tgt': drv_secc_color,
                    'ref_torso_img': ref_torso_imgs, 'bg_img': bg_img, 'segmap': segmaps,
                    'kp_s': kp_src, 'kp_d': kp_drv}
            camera = self.ds['cameras'][drv_idx]
            gen_output = self.secc2video_model.forward(img=None, camera=camera, cond=cond, ret={}, cache_backbone=False, use_cached_backbone=True)
            pred_img = gen_output['image']
            pred_img = ((pred_img.permute(0, 2, 3, 1) + 1)/2 * 255).int().cpu().numpy().astype(np.uint8)
            video_writer.append_data(pred_img[0])
        video_writer.close()
        self.model.train()
    def masked_error_loss(self, img_pred, img_gt, mask, unmasked_weight=0.1, mode='l1'):
        # 对raw图像，因为deform的原因背景没法全黑，导致这部分mse过高，我们将其mask掉，只计算人脸部分
        masked_weight = 1.0
        weight_mask = mask.float() * masked_weight + (~mask).float() * unmasked_weight
        if mode == 'l1':
            error = (img_pred - img_gt).abs().sum(dim=1) * weight_mask
        else:
            error = (img_pred - img_gt).pow(2).sum(dim=1) * weight_mask
        error.clamp_(0, max(0.5, error.quantile(0.8).item())) # clamp掉较高loss的pixel，避免姿态没对齐的pixel导致的异常值占主导影响训练
        loss = error.mean()
        return loss
    def dilate(self, bin_img, ksize=5, mode='max_pool'):
        """
        mode: max_pool or avg_pool
        """
        # bin_img, [1, h, w]
        pad = (ksize-1)//2
        bin_img = F.pad(bin_img, pad=[pad,pad,pad,pad], mode='reflect')
        if mode == 'max_pool':
            out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
        else:
            out = F.avg_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
        return out
        
    def dilate_mask(self, mask, ksize=21):
        mask = self.dilate(mask, ksize=ksize, mode='max_pool')
        return mask
    
    def set_unmasked_to_black(self, img, mask):
        out_img = img * mask.float() - (~mask).float() # -1 denotes black
        return out_img
    
    def dump_checkpoint(self, inp):
        checkpoint = {}
        # save optimizers
        optimizer_states = []
        self.optimizers = [self.optimizer]
        for i, optimizer in enumerate(self.optimizers):
            if optimizer is not None:
                state_dict = optimizer.state_dict()
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                optimizer_states.append(state_dict)
        checkpoint['optimizer_states'] = optimizer_states
        state_dict = {
            'model': self.model.state_dict(),
            'learnable_triplane': self.model.state_dict()['_last_cano_planes'],
        }
        del state_dict['model']['_last_cano_planes']
        checkpoint['state_dict'] = state_dict
        checkpoint['lora_args'] = self.lora_args
        person_ds = {}
        video_id = inp['video_id']
        img_name = f'data/processed/videos/{video_id}/gt_imgs/{format(0, "08d")}.jpg'
        gt_img = torch.tensor(cv2.resize(cv2.imread(img_name), (512, 512))[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
        person_ds['gt_img'] = gt_img.reshape([1, 3, 512, 512])
        person_ds['id'] = self.ds['id'].cpu().reshape([1, 80])
        person_ds['src_kp'] = self.ds['kps'][0].cpu()
        person_ds['video_id'] = inp['video_id']
        checkpoint['person_ds'] = person_ds
        return checkpoint
if __name__ == '__main__':
    import argparse, glob, tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--head_ckpt", default='') # checkpoints/0729_th1kh/secc_img2plane checkpoints/0720_img2planes/secc_img2plane_two_stage
    parser.add_argument("--torso_ckpt", default='checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig') # checkpoints/0729_th1kh/secc_img2plane checkpoints/0720_img2planes/secc_img2plane_two_stage
    parser.add_argument("--video_id", default='data/raw/examples/Trump_10s.mp4', help="identity source, we support (1) already processed <video_id> of GeneFace, (2) video path, (3) image path")
    parser.add_argument("--work_dir", default=None) 
    parser.add_argument("--max_updates", default=10000, type=int, help="for video, 2000 is good; for an image, 3~10 is good") 
    parser.add_argument("--test", action='store_true') 
    parser.add_argument("--batch_size", default=1, type=int, help="batch size during training, 1 needs 8GB, 2 needs 15GB") 
    parser.add_argument("--lr", default=0.001) 
    parser.add_argument("--lr_triplane", default=0.005, help="for video, 0.1; for an image, 0.001; for ablation with_triplane, 0.") 
    parser.add_argument("--lambda_lpips", default=0.2) 
    parser.add_argument("--lora_r", default=2, type=int, help="width of lora unit") 
    parser.add_argument("--lora_mode", default='secc2plane_sr', help='for video, full; for an image, none')
    args = parser.parse_args()
    inp = {
            'head_ckpt': args.head_ckpt,
            'torso_ckpt': args.torso_ckpt,
            'video_id': args.video_id,
            'work_dir': args.work_dir,
            'max_updates': args.max_updates,
            'batch_size': args.batch_size,
            'test': args.test,
            'lr': float(args.lr),
            'lr_triplane': float(args.lr_triplane),
            'lambda_lpips': float(args.lambda_lpips),
            'lora_mode': args.lora_mode,
            'lora_r': args.lora_r,
            }
    if inp['work_dir'] == None:
        video_id = os.path.basename(inp['video_id'])[:-4] if inp['video_id'].endswith((".mp4", ".png", ".jpg", ".jpeg")) else inp['video_id']
        inp['work_dir'] = f'checkpoints_mimictalk/{video_id}'
    os.makedirs(inp['work_dir'], exist_ok=True)
    trainer = LoRATrainer(inp)
    if inp['test']:
        trainer.test_loop(inp)
    else:
        trainer.training_loop(inp)
        trainer.test_loop(inp)
    print(" ")