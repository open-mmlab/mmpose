FROM openmmlab/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy1.3.1

# previous method of local installation fails so fallback to the official method
RUN pip install -U openmim

RUN mim install mmsegmentation
RUN python3 -m pip install ftfy regex

RUN git clone https://github.com/logivations/mmsegmentation.git /mmsegmentation

#MMPretrain
RUN mim install git+https://github.com/logivations/mmpretrain.git
RUN git clone https://github.com/logivations/mmpretrain.git /mmpretrain

RUN mim install mmpose
RUN git clone https://github.com/logivations/mmpose.git /mmpose

WORKDIR /root/workspace/mmdeploy
