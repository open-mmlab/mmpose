# Inference Speed

We summarize the model complexity and inference speed of major models in MMPose, including FLOPs, parameter counts and inference speeds on both CPU and GPU devices with different batch sizes. We also compare the mAP of different models on COCO human keypoint dataset, showing the trade-off between model performance and model complexity.

## Comparison Rules

To ensure the fairness of the comparison, the comparison experiments are conducted under the same hardware and software environment using the same dataset. We also list the mAP (mean average precision) on COCO human keypoint dataset of the models along with the corresponding config files.

For model complexity information measurement, we calculate the FLOPs and parameter counts of a model with corresponding input shape. Note that some layers or ops are currently not supported, for example, `DeformConv2d`, so you may need to check if all ops are supported and verify that the flops and parameter counts computation is correct.

For inference speed, we omit the time for data pre-processing and only measure the time for model forwarding and data post-processing. For each model setting, we keep the same data pre-processing methods to make sure the same feature input. We measure the inference speed on both CPU and GPU devices. For topdown heatmap models, we also test the case when the batch size is larger, e.g., 10, to test model performance in crowded scenes.

The inference speed is measured with frames per second (FPS), namely the average iterations per second, which can show how fast the model can handle an input. The higher, the faster, the better.

### Hardware

- GPU: GeForce GTX 1660 SUPER
- CPU: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz

### Software Environment

- Ubuntu 16.04
- Python 3.8
- PyTorch 1.10
- CUDA 10.2
- mmcv-full 1.3.17
- mmpose 0.20.0

## Model complexity information and inference speed results of major models in MMPose

| Algorithm | Model | config | Input size | mAP | Flops (GFLOPs) | Params (M) | GPU Inference Speed<br>(FPS)<sup>1</sup> | GPU Inference Speed<br>(FPS, bs=10)<sup>2</sup> | CPU Inference Speed<br>(FPS) | CPU Inference Speed<br>(FPS, bs=10) |
| :--- | :---------------: | :-----------------: |:--------------------: | :----------------------------: | :-----------------: | :---------------: |:--------------------: | :----------------------------: | :-----------------: | :-----------------: |
| topdown_heatmap | Alexnet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/alexnet_coco_256x192.py) | (3, 192, 256) | 0.397 | 1.42 | 5.62 | 229.21 ± 16.91 | 33.52 ± 1.14 | 13.92 ± 0.60 | 1.38 ± 0.02 |
| topdown_heatmap | CPM | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/cpm_coco_256x192.py) | (3, 192, 256) | 0.623 | 63.81 | 31.3 | 11.35 ± 0.22 | 3.87 ± 0.07 | 0.31 ± 0.01 | 0.03 ± 0.00 |
| topdown_heatmap | CPM | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/cpm_coco_384x288.py) | (3, 288, 384) | 0.65 | 143.57 | 31.3 | 7.09 ± 0.14 | 2.10 ± 0.05 | 0.14 ± 0.00 | 0.01 ± 0.00 |
| topdown_heatmap | Hourglass-52 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hourglass52_coco_256x256.py) | (3, 256, 256) | 0.726 | 28.67 | 94.85 | 25.50 ± 1.68 | 3.99 ± 0.07 | 0.92 ± 0.03 | 0.09 ± 0.00 |
| topdown_heatmap | Hourglass-52 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hourglass52_coco_384x384.py) | (3, 384, 384) | 0.746 | 64.5 | 94.85 | 14.74 ± 0.8 | 1.86 ± 0.06 | 0.43 ± 0.03 | 0.04 ± 0.00 |
| topdown_heatmap | HRNet-W32 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py) | (3, 192, 256) | 0.746 | 7.7 | 28.54 | 22.73 ± 1.12 | 6.60 ± 0.14 | 2.73 ± 0.11 | 0.32 ± 0.00 |
| topdown_heatmap | HRNet-W32 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_384x288.py) | (3, 288, 384) | 0.76 | 17.33 | 28.54 | 22.78 ± 1.21 | 3.28 ± 0.08 | 1.35 ± 0.05 | 0.14 ± 0.00 |
| topdown_heatmap | HRNet-W48 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py) | (3, 192, 256) | 0.756 | 15.77 | 63.6 | 22.01 ± 1.10 | 3.74 ± 0.10 | 1.46 ± 0.05 | 0.16 ± 0.00 |
| topdown_heatmap | HRNet-W48 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288.py) | (3, 288, 384) | 0.767 | 35.48 | 63.6 | 15.03 ± 1.03 | 1.80 ± 0.03 | 0.68 ± 0.02 | 0.07 ± 0.00 |
| topdown_heatmap | LiteHRNet-30 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/litehrnet_30_coco_256x192.py) | (3, 192, 256) | 0.675 | 0.42 | 1.76 | 11.86 ± 0.38 | 9.77 ± 0.23 | 5.84 ± 0.39 | 0.80 ± 0.00 |
| topdown_heatmap | LiteHRNet-30 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/litehrnet_30_coco_384x288.py) | (3, 288, 384) | 0.7 | 0.95 | 1.76 | 11.52 ± 0.39 | 5.18 ± 0.11 | 3.45 ± 0.22 | 0.37 ± 0.00 |
| topdown_heatmap | MobilenetV2 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/mobilenetv2_coco_256x192.py) | (3, 192, 256) | 0.646 | 1.59 | 9.57 | 91.82 ± 10.98 | 17.85 ± 0.32 | 10.44 ± 0.80 | 1.05 ± 0.01 |
| topdown_heatmap | MobilenetV2 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/mobilenetv2_coco_384x288.py) | (3, 288, 384) | 0.673 | 3.57 | 9.57 | 71.27 ± 6.82 | 8.00 ± 0.15  | 5.01 ± 0.32 | 0.46 ± 0.00 |
| topdown_heatmap | MSPN-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/mspn50_coco_256x192.py) | (3, 192, 256) | 0.723 | 5.11 | 25.11 | 59.65 ± 3.74 | 9.51 ± 0.15  | 3.98 ± 0.21 | 0.43 ± 0.00 |
| topdown_heatmap | 2xMSPN-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/2xmspn50_coco_256x192.py) | (3, 192, 256) | 0.754 | 11.35 | 56.8 | 30.64 ± 2.61 | 4.74 ± 0.12 | 1.85 ± 0.08 | 0.20 ± 0.00 |
| topdown_heatmap | 3xMSPN-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/3xmspn50_coco_256x192.py) | (3, 192, 256) | 0.758 | 17.59 | 88.49 | 20.90 ± 1.82 | 3.22 ± 0.08 | 1.23 ± 0.04 | 0.13 ± 0.00 |
| topdown_heatmap | 4xMSPN-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/4xmspn50_coco_256x192.py) | (3, 192, 256) | 0.764 | 23.82 | 120.18 | 15.79 ± 1.14  | 2.45 ± 0.05 | 0.90 ± 0.03 | 0.10 ± 0.00 |
| topdown_heatmap | ResNest-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest50_coco_256x192.py) | (3, 192, 256) | 0.721 | 6.73 | 35.93 | 48.36 ± 4.12 | 7.48 ± 0.13 | 3.00 ± 0.13 | 0.33 ± 0.00 |
| topdown_heatmap | ResNest-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest50_coco_384x288.py) | (3, 288, 384) | 0.737 | 15.14 | 35.93 | 30.30 ± 2.30 | 3.62 ± 0.09 | 1.43 ± 0.05 | 0.13 ± 0.00 |
| topdown_heatmap | ResNest-101 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest101_coco_256x192.py) | (3, 192, 256) | 0.725 | 10.38 | 56.61 | 29.21 ± 1.98 | 5.30 ± 0.12 | 2.01 ± 0.08 | 0.22 ± 0.00 |
| topdown_heatmap | ResNest-101 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest101_coco_384x288.py) | (3, 288, 384) | 0.746 | 23.36 | 56.61 | 19.02 ± 1.40 | 2.59 ± 0.05  | 0.97 ± 0.03 | 0.09 ± 0.00 |
| topdown_heatmap | ResNest-200 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest200_coco_256x192.py) | (3, 192, 256) | 0.732 | 17.5 | 78.54 | 16.11 ± 0.71 | 3.29 ± 0.07  | 1.33 ± 0.02 | 0.14 ± 0.00 |
| topdown_heatmap | ResNest-200 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest200_coco_384x288.py) | (3, 288, 384) | 0.754 | 39.37 | 78.54 | 11.48 ± 0.68 | 1.58 ± 0.02 | 0.63 ± 0.01 | 0.06 ± 0.00 |
| topdown_heatmap | ResNest-269 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest269_coco_256x192.py) | (3, 192, 256) | 0.738 | 22.45 | 119.27 | 12.02 ± 0.47 | 2.60 ± 0.05 | 1.03 ± 0.01 | 0.11 ± 0.00 |
| topdown_heatmap | ResNest-269 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest269_coco_384x288.py) | (3, 288, 384) | 0.755 | 50.5 | 119.27 | 8.82 ± 0.42  | 1.24 ± 0.02 | 0.49 ± 0.01 | 0.05 ± 0.00 |
| topdown_heatmap | ResNet-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py) | (3, 192, 256) | 0.718 | 5.46 | 34 | 64.23 ± 6.05 | 9.33 ± 0.21 | 4.00 ± 0.10 | 0.41 ± 0.00 |
| topdown_heatmap | ResNet-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_384x288.py) | (3, 288, 384) | 0.731 | 12.29 | 34 | 36.78 ± 3.05 | 4.48 ± 0.12 | 1.92 ± 0.04 | 0.19 ± 0.00 |
| topdown_heatmap | ResNet-101 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res101_coco_256x192.py) | (3, 192, 256) | 0.726 | 9.11 | 52.99 | 43.35 ± 4.36 | 6.44 ± 0.14 | 2.57 ± 0.05 | 0.27 ± 0.00 |
| topdown_heatmap | ResNet-101 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res101_coco_384x288.py) | (3, 288, 384) | 0.748 | 20.5 | 52.99 | 23.29 ± 1.83 | 3.12 ± 0.09 | 1.23 ± 0.03 | 0.11 ± 0.00 |
| topdown_heatmap | ResNet-152 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res152_coco_256x192.py) | (3, 192, 256) | 0.735 | 12.77 | 68.64 | 32.31 ± 2.84 | 4.88 ± 0.17 | 1.89 ± 0.03 | 0.20 ± 0.00 |
| topdown_heatmap | ResNet-152 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res152_coco_384x288.py) | (3, 288, 384) | 0.75 | 28.73 | 68.64 | 17.32 ± 1.17 | 2.40 ± 0.04 | 0.91 ± 0.01 | 0.08 ± 0.00 |
| topdown_heatmap | ResNetV1d-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d50_coco_256x192.py) | (3, 192, 256) | 0.722 | 5.7 | 34.02 | 63.44 ± 6.09 | 9.09 ± 0.10 | 3.82 ± 0.10 | 0.39 ± 0.00 |
| topdown_heatmap | ResNetV1d-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d50_coco_384x288.py) | (3, 288, 384) | 0.73 | 12.82 | 34.02 | 36.21 ± 3.10 | 4.30 ± 0.12 | 1.82 ± 0.04 | 0.16 ± 0.00 |
| topdown_heatmap | ResNetV1d-101 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d101_coco_256x192.py) | (3, 192, 256) | 0.731 | 9.35 | 53.01 | 41.48 ± 3.76 | 6.33 ± 0.15 | 2.48 ± 0.05 | 0.26 ± 0.00 |
| topdown_heatmap | ResNetV1d-101 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d101_coco_384x288.py) | (3, 288, 384) | 0.748 | 21.04 | 53.01 | 23.49 ± 1.76 | 3.07 ± 0.07 | 1.19 ± 0.02 | 0.11 ± 0.00 |
| topdown_heatmap | ResNetV1d-152 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d152_coco_256x192.py) | (3, 192, 256) | 0.737 | 13.01 | 68.65 | 31.96 ± 2.87 | 4.69 ± 0.18 | 1.87 ± 0.02 | 0.19 ± 0.00 |
| topdown_heatmap | ResNetV1d-152 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d152_coco_384x288.py) | (3, 288, 384) | 0.752 | 29.26 | 68.65 | 17.31 ± 1.13 | 2.32 ± 0.04 | 0.88 ± 0.01 | 0.08 ± 0.00 |
| topdown_heatmap | ResNext-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext50_coco_256x192.py) | (3, 192, 256) | 0.714 | 5.61 | 33.47 | 48.34 ± 3.85 | 7.66 ± 0.13 | 3.71 ± 0.10 | 0.37 ± 0.00 |
| topdown_heatmap | ResNext-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext50_coco_384x288.py) | (3, 288, 384) | 0.724 | 12.62 | 33.47 | 30.66 ± 2.38 | 3.64 ± 0.11 | 1.73 ± 0.03 | 0.15 ± 0.00 |
| topdown_heatmap | ResNext-101 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext101_coco_256x192.py) | (3, 192, 256) | 0.726 | 9.29 | 52.62 | 27.33 ± 2.35 | 5.09 ± 0.13 | 2.45 ± 0.04 | 0.25 ± 0.00 |
| topdown_heatmap | ResNext-101 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext101_coco_384x288.py) | (3, 288, 384) | 0.743 | 20.91 | 52.62 | 18.19 ± 1.38  | 2.42 ± 0.04 | 1.15 ± 0.01 | 0.10 ± 0.00 |
| topdown_heatmap | ResNext-152 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext152_coco_256x192.py) | (3, 192, 256) | 0.73 | 12.98 | 68.39 | 19.61 ± 1.61 | 3.80 ± 0.13 | 1.83 ± 0.02 | 0.18 ± 0.00 |
| topdown_heatmap | ResNext-152 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext152_coco_384x288.py) | (3, 288, 384) | 0.742 | 29.21 | 68.39 | 13.14 ± 0.75 | 1.82 ± 0.03 | 0.85 ± 0.01 | 0.08 ± 0.00 |
| topdown_heatmap | RSN-18 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/rsn18_coco_256x192.py) | (3, 192, 256) | 0.704 | 2.27 | 9.14 | 47.80 ± 4.50 | 13.68 ± 0.25 | 6.70 ± 0.28 | 0.70 ± 0.00 |
| topdown_heatmap | RSN-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/rsn50_coco_256x192.py) | (3, 192, 256) | 0.723 | 4.11 | 19.33 | 27.22 ± 1.61 | 8.81 ± 0.13 | 3.98 ± 0.12 | 0.45 ± 0.00 |
| topdown_heatmap | 2xRSN-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/2xrsn50_coco_256x192.py) | (3, 192, 256) | 0.745 | 8.29 | 39.26 | 13.88 ± 0.64 | 4.78 ± 0.13 | 2.02 ± 0.04 | 0.23 ± 0.00 |
| topdown_heatmap | 3xRSN-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/3xrsn50_coco_256x192.py) | (3, 192, 256) | 0.75 | 12.47 | 59.2 | 9.40 ± 0.32 | 3.37 ± 0.09 | 1.34 ± 0.03 | 0.15 ± 0.00 |
| topdown_heatmap | SCNet-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/scnet50_coco_256x192.py) | (3, 192, 256) | 0.728 | 5.31 | 34.01 | 40.76 ± 3.08 | 8.35 ± 0.19 | 3.82 ± 0.08 | 0.40 ± 0.00 |
| topdown_heatmap | SCNet-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/scnet50_coco_384x288.py) | (3, 288, 384) | 0.751 | 11.94 | 34.01 | 32.61 ± 2.97 | 4.19 ± 0.10 | 1.85 ± 0.03 | 0.17 ± 0.00 |
| topdown_heatmap | SCNet-101 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/scnet101_coco_256x192.py) | (3, 192, 256) | 0.733 | 8.51 | 53.01 | 24.28 ± 1.19 | 5.80 ± 0.13 | 2.49 ± 0.05 | 0.27 ± 0.00  |
| topdown_heatmap | SCNet-101 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/scnet101_coco_384x288.py) | (3, 288, 384) | 0.752 | 19.14 | 53.01 | 20.43 ± 1.76 | 2.91 ± 0.06 | 1.23 ± 0.02 | 0.12 ± 0.00 |
| topdown_heatmap | SeresNet-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet50_coco_256x192.py) | (3, 192, 256) | 0.728 | 5.47 | 36.53 | 54.83 ± 4.94 | 8.80 ± 0.12 | 3.85 ± 0.10 | 0.40 ± 0.00 |
| topdown_heatmap | SeresNet-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet50_coco_384x288.py) | (3, 288, 384) | 0.748 | 12.3 | 36.53 | 33.00 ± 2.67 | 4.26 ± 0.12 | 1.86 ± 0.04 | 0.17 ± 0.00 |
| topdown_heatmap | SeresNet-101 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet101_coco_256x192.py) | (3, 192, 256) | 0.734 | 9.13 | 57.77 | 33.90 ± 2.65 | 6.01 ± 0.13 | 2.48 ± 0.05 | 0.26 ± 0.00 |
| topdown_heatmap | SeresNet-101 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet101_coco_384x288.py) | (3, 288, 384) | 0.753 | 20.53 | 57.77 | 20.57 ± 1.57 | 2.96 ± 0.07 | 1.20 ± 0.02 | 0.11 ± 0.00 |
| topdown_heatmap | SeresNet-152 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet152_coco_256x192.py) | (3, 192, 256) | 0.73 | 12.79 | 75.26 | 24.25 ± 1.95 | 4.45 ± 0.10 | 1.82 ± 0.02 | 0.19 ± 0.00 |
| topdown_heatmap | SeresNet-152 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet152_coco_384x288.py) | (3, 288, 384) | 0.753 | 28.76 | 75.26 | 15.11 ± 0.99  | 2.25 ± 0.04 | 0.88 ± 0.01 | 0.08 ± 0.00 |
| topdown_heatmap | ShuffleNetV1 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/shufflenetv1_coco_256x192.py) | (3, 192, 256) | 0.585 | 1.35 | 6.94 | 80.79 ± 8.95 | 21.91 ± 0.46 | 11.84 ± 0.59 | 1.25 ± 0.01 |
| topdown_heatmap | ShuffleNetV1 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/shufflenetv1_coco_384x288.py) | (3, 288, 384) | 0.622 | 3.05 | 6.94 | 63.45 ± 5.21 | 9.84 ± 0.10 | 6.01 ± 0.31 | 0.57 ± 0.00 |
| topdown_heatmap | ShuffleNetV2 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/shufflenetv2_coco_256x192.py) | (3, 192, 256) | 0.599 | 1.37 | 7.55 | 82.36 ± 7.30 | 22.68 ± 0.53 | 12.40 ± 0.66 | 1.34 ± 0.02 |
| topdown_heatmap | ShuffleNetV2 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/shufflenetv2_coco_384x288.py) | (3, 288, 384) | 0.636 | 3.08 | 7.55 | 63.63 ± 5.72 | 10.47 ± 0.16 | 6.32 ± 0.28 | 0.63 ± 0.01  |
| topdown_heatmap | VGG16 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vgg16_bn_coco_256x192.py) | (3, 192, 256) | 0.698 | 16.22 | 18.92 | 51.91 ± 2.98 | 6.18 ± 0.13 | 1.64 ± 0.03 | 0.15 ± 0.00 |
| topdown_heatmap | VIPNAS + ResNet-50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vipnas_res50_coco_256x192.py) | (3, 192, 256) | 0.711 | 1.49 | 7.29 | 34.88 ± 2.45 | 10.29 ± 0.13 | 6.51 ± 0.17 | 0.65 ± 0.00 |
| topdown_heatmap | VIPNAS + MobileNetV3 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vipnas_mbv3_coco_256x192.py) | (3, 192, 256) | 0.7 | 0.76 | 5.9 | 53.62 ± 6.59 | 11.54 ± 0.18 | 1.26 ± 0.02 | 0.13 ± 0.00 |
| Associative Embedding | HigherHRNet-W32 | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_512x512.py) | (3, 512, 512) | 0.677 | 46.58 | 28.65 | 7.80 ± 0.67 | / | 0.28 ± 0.02 | / |
| Associative Embedding | HigherHRNet-W32 | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_640x640.py) | (3, 640, 640) | 0.686 | 72.77 | 28.65 | 5.30 ± 0.37 | / | 0.17 ± 0.01 | / |
| Associative Embedding | HigherHRNet-W48 | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512.py) | (3, 512, 512) | 0.686 | 96.17 | 63.83 | 4.55 ± 0.35 | / | 0.15 ± 0.01 | / |
| Associative Embedding | Hourglass-AE | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hourglass_ae_coco_512x512.py) | (3, 512, 512) | 0.613 | 221.58 | 138.86 | 3.55 ± 0.24 | / | 0.08 ± 0.00 | / |
| Associative Embedding | HRNet-W32 | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py) | (3, 512, 512) | 0.654 | 41.1 | 28.54 | 8.93 ± 0.76 | / | 0.33 ± 0.02 | / |
| Associative Embedding | HRNet-W48 | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w48_coco_512x512.py) | (3, 512, 512) | 0.665 | 84.12 | 63.6 | 5.27 ± 0.43 | / | 0.18 ± 0.01 | / |
| Associative Embedding | MobilenetV2 | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/mobilenetv2_coco_512x512.py) | (3, 512, 512) | 0.38 | 8.54 | 9.57 | 21.24 ± 1.34 | / | 0.81 ± 0.06 | / |
| Associative Embedding | ResNet-50 | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res50_coco_512x512.py) | (3, 512, 512) | 0.466 | 29.2 | 34 | 11.71 ± 0.97 | / | 0.41 ± 0.02 | / |
| Associative Embedding | ResNet-50 | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res50_coco_640x640.py) | (3, 640, 640) | 0.479 | 45.62 | 34 | 8.20 ± 0.58 | / | 0.26 ± 0.02 | / |
| Associative Embedding | ResNet-101 | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res101_coco_512x512.py) | (3, 512, 512) | 0.554 | 48.67 | 53 | 8.26 ± 0.68 | / | 0.28 ± 0.02 | / |
| Associative Embedding | ResNet-101 | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res152_coco_512x512.py) | (3, 512, 512) | 0.595 | 68.17 | 68.64 | 6.25 ± 0.53 | / | 0.21 ± 0.01 | / |
| DeepPose | ResNet-50 | [config](/configs/body/2d_kpt_sview_rgb_img/deeppose/coco/res50_coco_256x192.py) | (3, 192, 256) | 0.526 | 4.04 | 23.58 | 82.20 ± 7.54 | / | 5.50 ± 0.18 | / |
| DeepPose | ResNet-101 | [config](/configs/body/2d_kpt_sview_rgb_img/deeppose/coco/res101_coco_256x192.py) | (3, 192, 256) | 0.56 | 7.69 | 42.57 | 48.93 ± 4.02 | / | 3.10 ± 0.07 | / |
| DeepPose | ResNet-152 | [config](/configs/body/2d_kpt_sview_rgb_img/deeppose/coco/res152_coco_256x192.py) | (3, 192, 256) | 0.583 | 11.34 | 58.21 | 35.06 ± 3.50 | / | 2.19 ± 0.04 | / |

<sup>1</sup> Note that we run multiple iterations and record the time of each iteration, and the mean and standard deviation value of FPS are both shown.

<sup>2</sup> The FPS is defined as the average iterations per second, regardless of the batch size in this iteration.
