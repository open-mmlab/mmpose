# MMPose for AIGC (AI Generated Content)

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222403836-c65ba905-4bdd-4a44-834c-ff8d5959649d.png" width=1000 />
</div>

English | [简体中文](./README_CN.md)

This project will demonstrate how to use MMPose to generate skeleton images for pose guided AI image generation.

Currently, we support:

- [T2I Adapter](https://huggingface.co/spaces/Adapter/T2I-Adapter)

Please feel free to share interesting pose-guided AIGC projects to us!

## Get Started

### Step 1: Preparation

#### Download Pre-compiled MMdeploy and Build

```shell
# Download pre-compiled files
wget https://github.com/open-mmlab/mmdeploy/releases/download/v1.0.0rc3/mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1.tar.gz

# Unzip files
tar -xzvf mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1.tar.gz

# Go to the sdk folder
cd mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1/sdk

# Init environment
source env.sh

# If opencv 3+ is not installed on your system, execute the following command.
# If it is installed, skip this command
bash opencv.sh

# Compile executable programs
bash build.sh
```

#### Download Models

```shell
# Go to mmdeploy folder
cd ../

# Download models
wget https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-cpu.zip

# Unzip files
unzip rtmpose-cpu.zip
```

The files are organized as follows:

```shell
|----mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1
|    |----sdk
|    |----rtmpose-ort
|    |    |----rtmdet-nano
|    |    |----rtmpose-m
|    |    |----000000147979.jpg
|    |    |----t2i-adapter_skeleton.txt
```

### Step 2: Generate a Skeleton Image

```shell
./bin/pose_tracker \
    ./rtmpose-ort/rtmdet-nano/ \
    ./rtmpose-ort/rtmpose-m \
    ./rtmpose-ort/000000147979.jpg \
    --background black \
    --skeleton ./rtmpose-ort/t2i-adapter_skeleton.txt \
    --output ./skeleton_res.jpg \
    --pose_kpt_thr 0.4 \
    --show -1
```

**Arguments description:**

```shell
required arguments:
  det_model             Object detection model path [string]
  pose_model            Pose estimation model path [string]
  input                 Input video path or camera index [string]

optional arguments:
  --device              Device name, e.g. "cpu", "cuda" [string = "cpu"]
  --output              Output video path or format string [string = ""]
  --output_size         Long-edge of output frames [int32 = 0]
  --flip                Set to 1 for flipping the input horizontally [int32 = 0]
  --show                Delay passed to `cv::waitKey` when using `cv::imshow`;
                        -1: disable [int32 = 1]
  --skeleton            Path to skeleton data or name of predefined skeletons:
                        "coco", "coco-wholebody" [string = "coco"]
  --background          Output background, "default": original image, "black":
                        black background [string = "default"]
  --det_interval        Detection interval [int32 = 1]
  --det_label           Detection label use for pose estimation [int32 = 0]
                        (0 refers to 'person' in coco)
  --det_thr             Detection score threshold [double = 0.5]
  --det_min_bbox_size   Detection minimum bbox size [double = -1]
  --det_nms_thr         NMS IOU threshold for merging detected bboxes and
                        bboxes from tracked targets [double = 0.7]
  --pose_max_num_bboxes Max number of bboxes used for pose estimation per frame
                        [int32 = -1]
  --pose_kpt_thr        Threshold for visible key-points [double = 0.5]
  --pose_min_keypoints  Min number of key-points for valid poses, -1 indicates
                        ceil(n_kpts/2) [int32 = -1]
  --pose_bbox_scale     Scale for expanding key-points to bbox [double = 1.25]
  --pose_min_bbox_size  Min pose bbox size, tracks with bbox size smaller than
                        the threshold will be dropped [double = -1]
  --pose_nms_thr        NMS OKS/IOU threshold for suppressing overlapped poses,
                        useful when multiple pose estimations collapse to the
                        same target [double = 0.5]
  --track_iou_thr       IOU threshold for associating missing tracks
                        [double = 0.4]
  --track_max_missing   Max number of missing frames before a missing tracks is
                        removed [int32 = 10]
```

For more details, you can refer to [RTMPose](../rtmpose/README.md).

The input image and its skeleton are as follows:

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222318807-a0a2a87a-cfc4-46cc-b647-e8f8bc44cfe3.jpg" width=450 /><img src='https://user-images.githubusercontent.com/13503330/222318943-6dba5f52-158a-427a-8222-03628addc051.jpg' width=450/>
</div>

### Step 3: Upload to T2I-Adapter

The demo page of T2I- Adapter is [Here](https://huggingface.co/spaces/Adapter/T2I-Adapter).

[![Huggingface Gradio](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/ChongMou/T2I-Adapter)

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222319401-883daca0-ba99-4d21-850c-199aa7868e0f.png" width=900 />
</div>

For example:

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222319507-c8862ac3-43a9-4672-8f57-ae2f3c2834e6.png" width=900 />
</div>

## Gallery

> A lady with a fish

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222319618-ee71ea71-88e2-4b61-81a0-bf675e0f8fbc.jpg" width=280 height=300 /><img src="https://user-images.githubusercontent.com/13503330/222319709-c19bef9a-ff02-4b09-a499-afd24f62399b.png" width=280 height=300/>
</div>

> An astronaut riding a bike on the moon

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222318943-6dba5f52-158a-427a-8222-03628addc051.jpg" width=280 height=300 /><img src="https://user-images.githubusercontent.com/13503330/222319895-c753620e-9c02-49ea-8586-c021d50f7225.png" width=280 height=300/>
</div>

> An astronaut riding a bike on Mars

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222318943-6dba5f52-158a-427a-8222-03628addc051.jpg" width=280 height=300 /><img src="https://user-images.githubusercontent.com/13503330/222319949-e4b4f5ff-888e-4080-9c1e-9bcafd17f306.png" width=280 height=300/>
</div>

> An astronaut riding a bike on Jupiter

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222318943-6dba5f52-158a-427a-8222-03628addc051.jpg" width=280 height=300 /><img src="https://user-images.githubusercontent.com/13503330/222320017-ae7fe863-fda6-4dcc-b9fb-2a44609c170f.png" width=280 height=300/>
</div>

> Monkey king

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222318943-6dba5f52-158a-427a-8222-03628addc051.jpg" width=280 height=300 /><img src="https://user-images.githubusercontent.com/13503330/222341871-4beac696-7d51-490b-94b2-2e3f1adb6927.jpg" width=280 height=300/>
</div>
