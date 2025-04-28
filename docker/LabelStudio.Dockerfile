# It is important that cuda supports the video card architectures that are important to us:
# NVIDIA GeForce RTX 3060 / RTX 3060 Ti - sm_86
# NVIDIA GeForce RTX 2080 Ti            - sm_75
# NVIDIA A100-SXM4-40GB                 - sm_80

ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y \
    git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install xtcocotools
RUN pip install cython
RUN pip install xtcocotools

# Install MMEngine and MMCV
RUN pip install openmim
RUN mim install mmengine "mmcv>=2.0.0"

# Install MMPose
RUN conda clean --all
RUN git clone https://github.com/logivations/mmpose.git /mmpose

# Checkout to branch (TODO: Remove after merge)
WORKDIR /mmpose
RUN git checkout pallet_detection 
RUN git pull

# Install requirements
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
RUN mim install "mmdet>=3.1.0"
RUN pip install future tensorboard albumentations
RUN pip install setuptools==59.5.0
