# Benchmark

We compare our results with some popular frameworks and official releases in terms of speed and accuracy.

## Comparision Rules

Here we compare our MMPose repo with other pose estimation toolboxes in the same data and model settings.

To ensure the fairness of the comparison, the comparison experiments were conducted under the same hardware environment and using the same dataset.
For each model setting, we kept the same data pre-processing methods to make sure the same feature input.

The time we measured is the average training time for an iteration, including data processing and model training.
The training speed is measure with s/iter. The lower, the better.

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

We demonstrate the superiority of our MMPose framework in terms of speed and accuracy on the standard COCO keypoint detection benchmark.
The mAP (the mean average precision) is used as the evaluation metrics.

| Model | Input size| MMPose (s/iter) | HRNet (s/iter) | MMPose (mAP) | HRNet (mAP) |
| :--- | :---------------: | :---------------: |:--------------------: | :----------------------------: | :-----------------: |
| resnet_50  | 256x192  | **1.10** | 1.12 | **0.718** | 0.704 |
| resnet_50  | 384x288  | **1.83** | 2.15 | **0.731** | 0.722 |
| resnet_101 | 256x192  | **1.10** | 1.46 | **0.726** | 0.714 |
| resnet_101 | 384x288  | **1.89** | 3.08 | **xxxxx** | 0.736 |
| resnet_152 | 256x192  | **1.11** | 1.71 | **xxxxx** | 0.720 |
| resnet_152 | 384x288  | **1.85** | 3.71 | **0.750** | 0.743 |
| hrnet_w32  | 256x192  | **1.11** | 2.09 | **0.746** | 0.744 |
| hrnet_w32  | 384x288  | **1.70** | 3.36 | **0.760** | 0.758 |
| hrnet_w48  | 256x192  | **1.08** | 2.28 | **0.756** | 0.751 |
| hrnet_w48  | 384x288  | **1.84** | 3.46 | **0.767** | 0.763 |

## Hardware

- 8 NVIDIA Tesla V100 (32G) GPUs
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

## Software Environment

- Python 3.7
- PyTorch 1.4
- CUDA 10.1
- CUDNN 7.6.03
- NCCL 2.4.08
