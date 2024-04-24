# RTMPose3D: Real-Time 3D Pose Estimation toolkit based on RTMPose

## Abstract

RTMPose3D is a toolkit for real-time 3D pose estimation. It is based on the RTMPose model, which is a 2D pose estimation model that is capable of predicting 2D keypoints and body part associations in real-time. RTMPose3D extends RTMPose by adding a 3D pose estimation branch that can predict 3D keypoints from images directly.

## Usage

👉🏼 TRY RTMO NOW

python .\body3d_img2pose_demo.py .\configs\rtmdet_m_640-8xb32_coco-person.py ..\..\..\mmpose\checkpoints\rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth .\configs\rtmw3d-l_8xb64_cocktail14-384x288.py ..\..\..\mmpose\checkpoints\rtmw3d-l_cock14-0d4ad840_20240422.pth --input ..\..\tests\data\coco\000000000785.jpg --output-root results

```bash
cd /path/to/mmpose/projects/rtmpose3d
python body3d_img2pose_demo.py configs/rtmdet_m_640-8xb32_coco-person.py https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth configs\rtmw3d-l_8xb64_cocktail14-384x288.py rtmw3d-l_cock14-0d4ad840_20240422.pth --input /path/to/image --output-root /path/to/output
```
