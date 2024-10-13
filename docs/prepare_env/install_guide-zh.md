# 环境配置
[English Doc](./install_guide.md)

本文档陈述了搭建MimicTalk Python环境的步骤，我们使用了Conda来管理依赖（与`Real3D-Portrait`的依赖一致）。

以下配置已在 A100/V100 + CUDA12.1 中进行了验证。


# 安装Python依赖与CUDA
```bash
cd <MimicTalkRoot>
source <CondaRoot>/bin/activate
conda create -n mimictalk python=3.9
conda activate mimictalk

# MMCV for SegFormer network structure
# 其他依赖项
pip install -r docs/prepare_env/requirements.txt -v
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install cython
pip install openmim==0.3.9
mim install mmcv==2.1.0 # 使用mim来加速mmcv安装
## 从源代码build pytorch3d
## 这可能会花费较长时间（可能数十分钟左右）；由于要连接Github，可能经常面临time-out问题，请考虑使用代理。
# 安装pytorch3d之前, 需要安装CUDA-12.1 (https://developer.nvidia.com/cuda-toolkit-archive) 并确保 /usr/local/cuda 指向了 `cuda-12.1` 目录
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

