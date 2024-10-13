# Prepare the Environment
[中文文档](./install_guide-zh.md)

This guide is about building a python environment for MimicTalk with Conda (the same as `Real3D-Portrait`).

The following installation process is verified in A100/V100 + CUDA12.1.

# Install Python Packages & CUDA
```bash
cd <MimicTalkRoot>
source <CondaRoot>/bin/activate
conda create -n mimictalk python=3.9
conda activate mimictalk

# MMCV for SegFormer network structure
# other dependencies
pip install -r docs/prepare_env/requirements.txt -v
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install cython
pip install openmim==0.3.9
mim install mmcv==2.1.0 # use mim to speed up installation for mmcv
## build pytorch3d from Github's source code. 
## It may take a long time (maybe tens of minutes), Proxy is recommended if encountering the time-out problem
# Before install pytorch3d, you need to install CUDA-12.1 (https://developer.nvidia.com/cuda-toolkit-archive) and make sure /usr/local/cuda points to the `cuda-12.1` directory
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

```