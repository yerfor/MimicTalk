"""
用于推理 inference/train_mimictalk_on_a_video.py 得到的person-specific模型
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import librosa
import random
import time
import numpy as np
import importlib
import tqdm
import copy
import cv2

# common utils
from utils.commons.hparams import hparams, set_hparams
from utils.commons.tensor_utils import move_to_cuda, convert_to_tensor
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint
# 3DMM-related utils
from deep_3drecon.deep_3drecon_models.bfm import ParametricFaceModel
from data_util.face3d_helper import Face3DHelper
from data_gen.utils.process_image.fit_3dmm_landmark import fit_3dmm_for_a_image
from data_gen.utils.process_video.fit_3dmm_landmark import fit_3dmm_for_a_video
from deep_3drecon.secc_renderer import SECC_Renderer
from data_gen.eg3d.convert_to_eg3d_convention import get_eg3d_convention_camera_pose_intrinsic
# Face Parsing 
from data_gen.utils.mp_feature_extractors.mp_segmenter import MediapipeSegmenter
from data_gen.utils.process_video.extract_segment_imgs import inpaint_torso_job, extract_background
# other inference utils
from inference.infer_utils import mirror_index, load_img_to_512_hwc_array, load_img_to_normalized_512_bchw_tensor
from inference.infer_utils import smooth_camera_sequence, smooth_features_xd
from inference.edit_secc import blink_eye_for_secc, hold_eye_opened_for_secc
from inference.real3d_infer import GeneFace2Infer


class AdaptGeneFace2Infer(GeneFace2Infer):
    def __init__(self, audio2secc_dir, head_model_dir, torso_model_dir, device=None, **kwargs):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.audio2secc_dir = audio2secc_dir
        self.head_model_dir = head_model_dir
        self.torso_model_dir = torso_model_dir
        self.audio2secc_model = self.load_audio2secc(audio2secc_dir)
        self.secc2video_model = self.load_secc2video(head_model_dir, torso_model_dir)
        self.audio2secc_model.to(device).eval()
        self.secc2video_model.to(device).eval()
        self.seg_model = MediapipeSegmenter()
        self.secc_renderer = SECC_Renderer(512)
        self.face3d_helper = Face3DHelper(use_gpu=True, keypoint_mode='lm68')
        self.mp_face3d_helper = Face3DHelper(use_gpu=True, keypoint_mode='mediapipe')
        # self.camera_selector = KNearestCameraSelector()

    def load_secc2video(self, head_model_dir, torso_model_dir):
        if torso_model_dir != '':
            config_dir = torso_model_dir if os.path.isdir(torso_model_dir) else os.path.dirname(torso_model_dir)
            set_hparams(f"{config_dir}/config.yaml", print_hparams=False)
            hparams['htbsr_head_threshold'] = 1.0
            self.secc2video_hparams = copy.deepcopy(hparams)
            ckpt = get_last_checkpoint(torso_model_dir)[0]
            lora_args = ckpt.get("lora_args", None)
            from modules.real3d.secc_img2plane_torso import OSAvatarSECC_Img2plane_Torso
            model = OSAvatarSECC_Img2plane_Torso(self.secc2video_hparams, lora_args=lora_args)
            load_ckpt(model, f"{torso_model_dir}", model_name='model', strict=True)
            self.learnable_triplane = nn.Parameter(torch.zeros([1, 3, model.triplane_hid_dim*model.triplane_depth, 256, 256]).float().cuda(), requires_grad=True)
            load_ckpt(self.learnable_triplane, f"{torso_model_dir}", model_name='learnable_triplane', strict=True)
            model._last_cano_planes = self.learnable_triplane
            if head_model_dir != '':
                print("| Warning: Assigned --torso_ckpt which also contains head, but --head_ckpt is also assigned, skipping the --head_ckpt.")
        else:
            from modules.real3d.secc_img2plane_torso import OSAvatarSECC_Img2plane
            set_hparams(f"{head_model_dir}/config.yaml", print_hparams=False)
            ckpt = get_last_checkpoint(head_model_dir)[0]
            lora_args = ckpt.get("lora_args", None)
            self.secc2video_hparams = copy.deepcopy(hparams)
            model = OSAvatarSECC_Img2plane(self.secc2video_hparams, lora_args=lora_args)
            load_ckpt(model, f"{head_model_dir}", model_name='model', strict=True)
            self.learnable_triplane = nn.Parameter(torch.zeros([1, 3, model.triplane_hid_dim*model.triplane_depth, 256, 256]).float().cuda(), requires_grad=True)
            model._last_cano_planes = self.learnable_triplane
            load_ckpt(model._last_cano_planes, f"{head_model_dir}", model_name='learnable_triplane', strict=True)
        self.person_ds = ckpt['person_ds']
        return model

    def prepare_batch_from_inp(self, inp):
        """
        :param inp: {'audio_source_name': (str)}
        :return: a dict that contains the condition feature of NeRF
        """
        sample = {}
        # Process Driving Motion
        if inp['drv_audio_name'][-4:] in ['.wav', '.mp3']:
            self.save_wav16k(inp['drv_audio_name'])
            if self.audio2secc_hparams['audio_type'] == 'hubert':
                hubert = self.get_hubert(self.wav16k_name)
            elif self.audio2secc_hparams['audio_type'] == 'mfcc':
                hubert = self.get_mfcc(self.wav16k_name) / 100

            f0 = self.get_f0(self.wav16k_name)
            if f0.shape[0] > len(hubert):
                f0 = f0[:len(hubert)]
            else:
                num_to_pad = len(hubert) - len(f0)
                f0 = np.pad(f0, pad_width=((0,num_to_pad), (0,0)))
            t_x = hubert.shape[0]
            x_mask = torch.ones([1, t_x]).float() # mask for audio frames
            y_mask = torch.ones([1, t_x//2]).float() # mask for motion/image frames
            sample.update({
                'hubert': torch.from_numpy(hubert).float().unsqueeze(0).cuda(),
                'f0': torch.from_numpy(f0).float().reshape([1,-1]).cuda(),
                'x_mask': x_mask.cuda(),
                'y_mask': y_mask.cuda(),
                })
            sample['blink'] = torch.zeros([1, t_x, 1]).long().cuda()
            sample['audio'] = sample['hubert']
            sample['eye_amp'] = torch.ones([1, 1]).cuda() * 1.0
        elif inp['drv_audio_name'][-4:] in ['.mp4']:
            drv_motion_coeff_dict = fit_3dmm_for_a_video(inp['drv_audio_name'], save=False)
            drv_motion_coeff_dict = convert_to_tensor(drv_motion_coeff_dict)
            t_x = drv_motion_coeff_dict['exp'].shape[0] * 2
            self.drv_motion_coeff_dict = drv_motion_coeff_dict
        elif inp['drv_audio_name'][-4:] in ['.npy']:
            drv_motion_coeff_dict = np.load(inp['drv_audio_name'], allow_pickle=True).tolist()
            drv_motion_coeff_dict = convert_to_tensor(drv_motion_coeff_dict)
            t_x = drv_motion_coeff_dict['exp'].shape[0] * 2
            self.drv_motion_coeff_dict = drv_motion_coeff_dict

        # Face Parsing
        sample['ref_gt_img'] = self.person_ds['gt_img'].cuda()
        img = self.person_ds['gt_img'].reshape([3, 512, 512]).permute(1, 2, 0)
        img = (img + 1) * 127.5
        img = np.ascontiguousarray(img.int().numpy()).astype(np.uint8)
        segmap = self.seg_model._cal_seg_map(img)
        sample['segmap'] = torch.tensor(segmap).float().unsqueeze(0).cuda()
        head_img = self.seg_model._seg_out_img_with_segmap(img, segmap, mode='head')[0]
        sample['ref_head_img'] = ((torch.tensor(head_img) - 127.5)/127.5).float().unsqueeze(0).permute(0, 3, 1,2).cuda() # [b,c,h,w]
        inpaint_torso_img, _, _, _ = inpaint_torso_job(img, segmap)
        sample['ref_torso_img'] = ((torch.tensor(inpaint_torso_img) - 127.5)/127.5).float().unsqueeze(0).permute(0, 3, 1,2).cuda() # [b,c,h,w]
        
        if inp['bg_image_name'] == '':
            bg_img = extract_background([img], [segmap], 'knn')
        else:
            bg_img = cv2.imread(inp['bg_image_name'])
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = cv2.resize(bg_img, (512,512))
        sample['bg_img'] = ((torch.tensor(bg_img) - 127.5)/127.5).float().unsqueeze(0).permute(0, 3, 1,2).cuda() # [b,c,h,w]

        # 3DMM, get identity code and camera pose
        image_name = f"data/raw/val_imgs/{self.person_ds['video_id']}_img.png"
        os.makedirs(os.path.dirname(image_name), exist_ok=True)
        cv2.imwrite(image_name, img[:,:,::-1])
        coeff_dict = fit_3dmm_for_a_image(image_name, save=False)
        coeff_dict['id'] = self.person_ds['id'].reshape([1,80]).numpy()

        assert coeff_dict is not None
        src_id = torch.tensor(coeff_dict['id']).reshape([1,80]).cuda()
        src_exp = torch.tensor(coeff_dict['exp']).reshape([1,64]).cuda()
        src_euler = torch.tensor(coeff_dict['euler']).reshape([1,3]).cuda()
        src_trans = torch.tensor(coeff_dict['trans']).reshape([1,3]).cuda()
        sample['id'] = src_id.repeat([t_x//2,1])

        # get the src_kp for torso model
        sample['src_kp'] = self.person_ds['src_kp'].cuda().reshape([1, 68, 3]).repeat([t_x//2,1,1])[..., :2] # [B, 68, 2]

        # get camera pose file
        random.seed(time.time())
        if inp['drv_pose_name'] in ['nearest', 'topk']:
            camera_ret = get_eg3d_convention_camera_pose_intrinsic({'euler': torch.tensor(coeff_dict['euler']).reshape([1,3]), 'trans': torch.tensor(coeff_dict['trans']).reshape([1,3])})
            c2w, intrinsics = camera_ret['c2w'], camera_ret['intrinsics']
            camera = np.concatenate([c2w.reshape([1,16]), intrinsics.reshape([1,9])], axis=-1)
            coeff_names, distance_matrix = self.camera_selector.find_k_nearest(camera, k=100)
            coeff_names = coeff_names[0] # squeeze
            if inp['drv_pose_name'] == 'nearest':
                inp['drv_pose_name'] = coeff_names[0]
            else:
                inp['drv_pose_name'] = random.choice(coeff_names)
            # inp['drv_pose_name'] = coeff_names[0]
        elif inp['drv_pose_name'] == 'random':
            inp['drv_pose_name'] = self.camera_selector.random_select()
        else:
            inp['drv_pose_name'] = inp['drv_pose_name']

        print(f"| To extract pose from {inp['drv_pose_name']}")

        # extract camera pose 
        if inp['drv_pose_name'] == 'static':
            sample['euler'] = torch.tensor(coeff_dict['euler']).reshape([1,3]).cuda().repeat([t_x//2,1]) # default static pose
            sample['trans'] = torch.tensor(coeff_dict['trans']).reshape([1,3]).cuda().repeat([t_x//2,1])
        else: # from file
            if inp['drv_pose_name'].endswith('.mp4'):
                # extract coeff from video
                drv_pose_coeff_dict = fit_3dmm_for_a_video(inp['drv_pose_name'], save=False)
            else:
                # load from npy
                drv_pose_coeff_dict = np.load(inp['drv_pose_name'], allow_pickle=True).tolist()
            print(f"| Extracted pose from {inp['drv_pose_name']}")
            eulers = convert_to_tensor(drv_pose_coeff_dict['euler']).reshape([-1,3]).cuda()
            trans = convert_to_tensor(drv_pose_coeff_dict['trans']).reshape([-1,3]).cuda()
            len_pose = len(eulers)
            index_lst = [mirror_index(i, len_pose) for i in range(t_x//2)]
            sample['euler'] = eulers[index_lst]
            sample['trans'] = trans[index_lst]

        # fix the z axis
        sample['trans'][:, -1] = sample['trans'][0:1, -1].repeat([sample['trans'].shape[0]])

        # mapping to the init pose
        if inp.get("map_to_init_pose", 'False') == 'True':
            diff_euler = torch.tensor(coeff_dict['euler']).reshape([1,3]).cuda() - sample['euler'][0:1]
            sample['euler'] = sample['euler'] + diff_euler
            diff_trans = torch.tensor(coeff_dict['trans']).reshape([1,3]).cuda() - sample['trans'][0:1]
            sample['trans'] = sample['trans'] + diff_trans

        # prepare camera
        camera_ret = get_eg3d_convention_camera_pose_intrinsic({'euler':sample['euler'].cpu(), 'trans':sample['trans'].cpu()})
        c2w, intrinsics = camera_ret['c2w'], camera_ret['intrinsics']
        # smooth camera
        camera_smo_ksize = 7
        camera = np.concatenate([c2w.reshape([-1,16]), intrinsics.reshape([-1,9])], axis=-1)
        camera = smooth_camera_sequence(camera, kernel_size=camera_smo_ksize) # [T, 25]
        camera = torch.tensor(camera).cuda().float()
        sample['camera'] = camera

        return sample

    @torch.no_grad()
    def forward_secc2video(self, batch, inp=None):
        num_frames = len(batch['drv_secc'])
        camera = batch['camera']
        src_kps = batch['src_kp']
        drv_kps = batch['drv_kp']
        cano_secc_color = batch['cano_secc']
        src_secc_color = batch['src_secc']
        drv_secc_colors = batch['drv_secc']
        ref_img_gt = batch['ref_gt_img']
        ref_img_head = batch['ref_head_img']
        ref_torso_img = batch['ref_torso_img']
        bg_img = batch['bg_img']
        segmap = batch['segmap']
        
        # smooth torso drv_kp
        torso_smo_ksize = 7
        drv_kps = smooth_features_xd(drv_kps.reshape([-1, 68*2]), kernel_size=torso_smo_ksize).reshape([-1, 68, 2])

        # forward renderer
        img_raw_lst = []
        img_lst = []
        depth_img_lst = []
        with torch.no_grad():
            for i in tqdm.trange(num_frames, desc="MimicTalk is rendering frames"):
                kp_src = torch.cat([src_kps[i:i+1].reshape([1, 68, 2]), torch.zeros([1, 68,1]).to(src_kps.device)],dim=-1)
                kp_drv = torch.cat([drv_kps[i:i+1].reshape([1, 68, 2]), torch.zeros([1, 68,1]).to(drv_kps.device)],dim=-1)
                cond={'cond_cano': cano_secc_color,'cond_src': src_secc_color, 'cond_tgt': drv_secc_colors[i:i+1].cuda(),
                        'ref_torso_img': ref_torso_img, 'bg_img': bg_img, 'segmap': segmap,
                        'kp_s': kp_src, 'kp_d': kp_drv}

                ########################################################################################################
                ### 相比real3d_infer只修改了这行👇，即cano_triplane来自cache里的learnable_triplane,而不是img预测的plane ####
                ########################################################################################################
                gen_output = self.secc2video_model.forward(img=None, camera=camera[i:i+1], cond=cond, ret={}, cache_backbone=False, use_cached_backbone=True)
                
                img_lst.append(gen_output['image'])
                img_raw_lst.append(gen_output['image_raw'])
                depth_img_lst.append(gen_output['image_depth'])

        # save demo video
        depth_imgs = torch.cat(depth_img_lst)
        imgs = torch.cat(img_lst)
        imgs_raw = torch.cat(img_raw_lst)
        secc_img = torch.cat([torch.nn.functional.interpolate(drv_secc_colors[i:i+1], (512,512)) for i in range(num_frames)])
        
        if inp['out_mode'] == 'concat_debug':
            secc_img = secc_img.cpu()
            secc_img = ((secc_img + 1) * 127.5).permute(0, 2, 3, 1).int().numpy()

            depth_img = F.interpolate(depth_imgs, (512,512)).cpu()
            depth_img = depth_img.repeat([1,3,1,1])
            depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
            depth_img = depth_img * 2 - 1
            depth_img = depth_img.clamp(-1,1)

            secc_img = secc_img / 127.5 - 1
            secc_img = torch.from_numpy(secc_img).permute(0, 3, 1, 2)
            imgs = torch.cat([ref_img_gt.repeat([imgs.shape[0],1,1,1]).cpu(), secc_img, F.interpolate(imgs_raw, (512,512)).cpu(), depth_img, imgs.cpu()], dim=-1)
        elif inp['out_mode'] == 'final':
            imgs = imgs.cpu()
        elif inp['out_mode'] == 'debug':
            raise NotImplementedError("to do: save separate videos")
        imgs = imgs.clamp(-1,1)

        import imageio
        import uuid
        debug_name = f'{uuid.uuid1()}.mp4'
        out_imgs = ((imgs.permute(0, 2, 3, 1) + 1)/2 * 255).int().cpu().numpy().astype(np.uint8)
        writer = imageio.get_writer(debug_name, fps=25, format='FFMPEG', codec='h264')
        for i in tqdm.trange(len(out_imgs), desc="Imageio is saving video"):
            writer.append_data(out_imgs[i])
        writer.close()
        
        out_fname = 'infer_out/tmp/' + os.path.basename(inp['drv_pose_name'])[:-4] + '.mp4' if inp['out_name'] == '' else inp['out_name']
        try:
            os.makedirs(os.path.dirname(out_fname), exist_ok=True)
        except: pass
        if inp['drv_audio_name'][-4:] in ['.wav', '.mp3']:
            # os.system(f"ffmpeg -i {debug_name} -i {inp['drv_audio_name']} -y -v quiet -shortest {out_fname}")
            cmd = f"/usr/bin/ffmpeg -i {debug_name} -i {self.wav16k_name} -y -r 25 -ar 16000 -c:v copy -c:a libmp3lame -pix_fmt yuv420p -b:v 2000k  -strict experimental -shortest {out_fname}"
            os.system(cmd)
            os.system(f"rm {debug_name}")
        else:
            ret = os.system(f"ffmpeg -i {debug_name} -i {inp['drv_audio_name']} -map 0:v -map 1:a -y -v quiet -shortest {out_fname}")
            if ret != 0: # 没有成功从drv_audio_name里面提取到音频, 则直接输出无音频轨道的纯视频
                os.system(f"mv {debug_name} {out_fname}")
        print(f"Saved at {out_fname}")
        return out_fname

if __name__ == '__main__':
    import argparse, glob, tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2m_ckpt", default='checkpoints/240112_icl_audio2secc_vox2_cmlr') # checkpoints/0727_audio2secc/audio2secc_withlm2d100_randomframe
    parser.add_argument("--head_ckpt", default='') # checkpoints/0729_th1kh/secc_img2plane checkpoints/0720_img2planes/secc_img2plane_two_stage
    parser.add_argument("--torso_ckpt", default='checkpoints_mimictalk/German_20s') 
    parser.add_argument("--bg_img", default='') # data/raw/val_imgs/bg3.png
    parser.add_argument("--drv_aud", default='data/raw/examples/80_vs_60_10s.wav')
    parser.add_argument("--drv_pose", default='data/raw/examples/German_20s.mp4') # nearest | topk | random | static | vid_name
    parser.add_argument("--drv_style", default='data/raw/examples/angry.mp4') # nearest | topk | random | static | vid_name
    parser.add_argument("--blink_mode", default='period') # none | period
    parser.add_argument("--temperature", default=0.3, type=float) # nearest | random
    parser.add_argument("--denoising_steps", default=20, type=int) # nearest | random
    parser.add_argument("--cfg_scale", default=1.5, type=float) # nearest | random
    parser.add_argument("--out_name", default='') # nearest | random
    parser.add_argument("--out_mode", default='concat_debug') # concat_debug | debug | final 
    parser.add_argument("--hold_eye_opened", default='False') # concat_debug | debug | final 
    parser.add_argument("--map_to_init_pose", default='True') # concat_debug | debug | final 
    parser.add_argument("--seed", default=None, type=int) # random seed, default None to use time.time()
 
    args = parser.parse_args()

    inp = {
            'a2m_ckpt': args.a2m_ckpt,
            'head_ckpt': args.head_ckpt,
            'torso_ckpt': args.torso_ckpt,
            'bg_image_name': args.bg_img,
            'drv_audio_name': args.drv_aud,
            'drv_pose_name': args.drv_pose,
            'drv_talking_style_name': args.drv_style,
            'blink_mode': args.blink_mode,
            'temperature': args.temperature,
            'denoising_steps': args.denoising_steps,
            'cfg_scale': args.cfg_scale,
            'out_name': args.out_name,
            'out_mode': args.out_mode,
            'map_to_init_pose': args.map_to_init_pose,
            'hold_eye_opened': args.hold_eye_opened,
            'seed': args.seed,
            }
    AdaptGeneFace2Infer.example_run(inp)