ARG PYTORCH="1.8.1"
ARG CUDA="10.2"
ARG CUDNN="7"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ARG MMCV="2.0.0rc4"
ARG MMPOSE="1.0.0rc1"

ENV PYTHONUNBUFFERED TRUE

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    openjdk-11-jre-headless \
    # MMDet Requirements
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:$PATH"
RUN export FORCE_CUDA=1


# MMLAB
ARG PYTORCH
ARG CUDA
RUN pip install mmengine
RUN ["/bin/bash", "-c", "pip install mmcv==${MMCV}} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]
RUN pip install mmpose==${MMPOSE}

# TORCHSEVER
RUN pip install torchserve torch-model-archiver

RUN useradd -m model-server \
    && mkdir -p /home/model-server/tmp

COPY entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh \
    && chown -R model-server /home/model-server

COPY config.properties /home/model-server/config.properties
RUN mkdir /home/model-server/model-store && chown -R model-server /home/model-server/model-store

EXPOSE 8080 8081 8082

USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
