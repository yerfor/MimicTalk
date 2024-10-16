# MimicTalk: Mimicking a personalized and expressive 3D talking face in few minutes | NeurIPS 2024
[![arXiv](https://img.shields.io/badge/arXiv-Paper-%3CCOLOR%3E.svg)](https://arxiv.org/abs/2401.08503)| [![GitHub Stars](https://img.shields.io/github/stars/yerfor/MimicTalk
)](https://github.com/yerfor/MimicTalk)  | [English Readme](./README.md)

这个仓库是MimicTalk的官方PyTorch实现, 用于实现特定说话人的高表现力的虚拟人视频合成。该仓库代码基于我们先前的工作[Real3D-Portrait](https://github.com/yerfor/Real3DPortrait) (ICLR 2024)，即基于NeRF的one-shot说话人合成，这让Mimictalk的训练加速且效果增强。您可以访问我们的[项目页面](https://mimictalk.github.io/)以观看Demo视频, 阅读我们的[论文](https://arxiv.org/abs/2410.06734)以了解技术细节。

<p align="center">
    <br>
    <img src="assets/mimictalk.png" width="100%"/>
    <br>
</p>

# 快速上手！
## 安装环境
请参照[环境配置文档](docs/prepare_env/install_guide-zh.md)，配置Conda环境`mimictalk`
## 下载预训练与第三方模型
### 3DMM BFM模型
下载3DMM BFM模型：[Google Drive](https://drive.google.com/drive/folders/1o4t5YIw7w4cMUN4bgU9nPf6IyWVG1bEk?usp=sharing) 或 [BaiduYun Disk](https://pan.baidu.com/s/1aqv1z_qZ23Vp2VP4uxxblQ?pwd=m9q5 ) 提取码: m9q5


下载完成后，放置全部的文件到`deep_3drecon/BFM`里，文件结构如下：
```
deep_3drecon/BFM/
├── 01_MorphableModel.mat
├── BFM_exp_idx.mat
├── BFM_front_idx.mat
├── BFM_model_front.mat
├── Exp_Pca.bin
├── facemodel_info.mat
├── index_mp468_from_mesh35709.npy
├── mediapipe_in_bfm53201.npy
└── std_exp.txt
```

### 预训练模型
下载预训练的MimicTalk相关Checkpoints：[Google Drive](https://drive.google.com/drive/folders/1Kc6ueDO9HFDN3BhtJCEKNCZtyKHSktaA?usp=sharing) or [BaiduYun Disk](https://pan.baidu.com/s/1nQKyGV5JB6rJtda7qsThUg?pwd=mimi) 提取码: mimi
  
下载完成后，放置全部的文件到`checkpoints`与`checkpoints_mimictalk`里并解压，文件结构如下：
```
checkpoints/
├── mimictalk_orig
│   └── os_secc2plane_torso
│       ├── config.yaml
│       └── model_ckpt_steps_100000.ckpt
|-- 240112_icl_audio2secc_vox2_cmlr
│     ├── config.yaml
│     └── model_ckpt_steps_1856000.ckpt
└── pretrained_ckpts
    └── mit_b0.pth

checkpoints_mimictalk/
└── German_20s
    ├── config.yaml
    └── model_ckpt_steps_10000.ckpt
```

## MimicTalk训练与推理的最简命令
```
python inference/train_mimictalk_on_a_video.py # train the model, this may take 10 minutes for 2,000 steps
python inference/mimictalk_infer.py # infer the model
```


# 训练与推理细节
我们目前提供了**命令行（CLI）**与**Gradio WebUI**推理方式。音频驱动推理的人像信息来自于`torso_ckpt`，因此需要至少再提供`driving audio`用于推理。另外，可以提供`style video`让模型能够预测与该视频风格一致的说话人动作。

首先，切换至项目根目录并启用Conda环境：
```bash
cd <Real3DPortraitRoot>
conda activate mimictalk
export PYTHONPATH=./
export HF_ENDPOINT=https://hf-mirror.com
```

## Gradio WebUI推理
启动Gradio WebUI，按照提示上传素材，点击`Training`按钮进行训练；训练完成后点击`Generate`按钮即可推理：
```bash
python inference/app_mimictalk.py
```

## 命令行特定说话人训练

需要至少提供`source video`，训练指令：
```bash
python inference/train_mimictalk_on_a_video.py \
--video_id <PATH_TO_SOURCE_VIDEO> \
--max_updates <UPDATES_NUMBER> \
--work_dir <PATH_TO_SAVING_CKPT>
```

一些可选参数注释：

- `--torso_ckpt` 预训练的Real3D-Portrait模型
- `--max_updates` 训练更新次数
- `--batch_size` 训练的batch size： `1` 需要约8GB显存; `2`需要约15GB显存
- `--lr_triplane` triplane的学习率：对于视频输入,  应为0.1; 对于图片输入，应为0.001
- `--work_dir` 未指定时，将默认存储在`checkpoints_mimictalk/`中

指令示例：
```bash
python inference/train_mimictalk_on_a_video.py \
--video_id data/raw/videos/German_20s.mp4 \
--max_updates 2000 \
--work_dir checkpoints_mimictalk/German_20s
```

## 命令行推理

需要至少提供`driving audio`，可选提供`driving style`，推理指令：
```bash
python inference/mimictalk_infer.py \
--drv_aud <PATH_TO_AUDIO> \
--drv_style <PATH_TO_STYLE_VIDEO, OPTIONAL> \
--drv_pose <PATH_TO_POSE_VIDEO, OPTIONAL> \
--bg_img <PATH_TO_BACKGROUND_IMAGE, OPTIONAL> \
--out_name <PATH_TO_OUTPUT_VIDEO, OPTIONAL>
```

一些可选参数注释：
- `--drv_pose` 指定时提供了运动pose信息，不指定则为静态运动
- `--bg_img` 指定时提供了背景信息，不指定则为source image提取的背景
- `--mouth_amp` 嘴部张幅参数，值越大张幅越大
- `--map_to_init_pose` 值为`True`时，首帧的pose将被映射到source pose，后续帧也作相同变换
- `--temperature` 代表audio2motion的采样温度，值越大结果越多样，但同时精确度越低
- `--out_name` 不指定时，结果将保存在`infer_out/tmp/`中
- `--out_mode` 值为`final`时，只输出说话人视频；值为`concat_debug`时，同时输出一些可视化的中间结果

推理命令例子：
```bash
python inference/mimictalk_infer.py \
--drv_aud data/raw/examples/Obama_5s.wav \
--drv_pose data/raw/examples/German_20s.mp4 \
--drv_style data/raw/examples/German_20s.mp4 \
--bg_img data/raw/examples/bg.png \
--out_name output.mp4 \
--out_mode final
```

# 声明
任何组织或个人未经本人同意，不得使用本文提及的任何技术生成他人说话的视频，包括但不限于政府领导人、政界人士、社会名流等。如不遵守本条款，则可能违反版权法。

# 引用我们
如果这个仓库对你有帮助，请考虑引用我们的工作：
```
@inproceedings{ye2024mimicktalk,
    author    = {Ye, Zhenhui and Zhong, Tianyun and Ren, Yi and Yang, Jiaqi and Li, Weichuang and Huang, Jiangwei and Jiang, Ziyue and He, Jinzheng and Huang, Rongjie and Liu, Jinglin and Zhang, Chen and Yin, Xiang and Ma, Zejun and Zhao, Zhou},
    title     = {MimicTalk: Mimicking a personalized and expressive 3D talking face in few minutes},
    journal   = {NeurIPS},
    year      = {2024},
}
@inproceedings{ye2024real3d,
  title = {Real3D-Portrait: One-shot Realistic 3D Talking Portrait Synthesis},
  author = {Ye, Zhenhui and Zhong, Tianyun and Ren, Yi and Yang, Jiaqi and Li, Weichuang and Huang, Jiawei and Jiang, Ziyue and He, Jinzheng and Huang, Rongjie and Liu, Jinglin and others},
  journal  = {ICLR},
  year={2024}
}
```