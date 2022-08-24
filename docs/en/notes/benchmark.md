# Benchmark

We compare our results with some popular frameworks and official releases in terms of speed and accuracy.

## Comparison Rules

Here we compare our MMPose repo with other pose estimation toolboxes in the same data and model settings.

To ensure the fairness of the comparison, the comparison experiments were conducted under the same hardware environment and using the same dataset.
For each model setting, we kept the same data pre-processing methods to make sure the same feature input.
In addition, we also used Memcached, a distributed memory-caching system, to load the data in all the compared toolboxes.
This minimizes the IO time during benchmark.

The time we measured is the average training time for an iteration, including data processing and model training.
The training speed is measure with s/iter. The lower, the better.

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

We demonstrate the superiority of our MMPose framework in terms of speed and accuracy on the standard COCO keypoint detection benchmark.
The mAP (the mean average precision) is used as the evaluation metric.

| Model      | Input size | MMPose (s/iter) | HRNet (s/iter) | MMPose (mAP) | HRNet (mAP) |
| :--------- | :--------: | :-------------: | :------------: | :----------: | :---------: |
| resnet_50  |  256x192   |    **0.28**     |      0.64      |  **0.718**   |    0.704    |
| resnet_50  |  384x288   |    **0.81**     |      1.24      |  **0.731**   |    0.722    |
| resnet_101 |  256x192   |    **0.36**     |      0.84      |  **0.726**   |    0.714    |
| resnet_101 |  384x288   |    **0.79**     |      1.53      |  **0.748**   |    0.736    |
| resnet_152 |  256x192   |    **0.49**     |      1.00      |  **0.735**   |    0.720    |
| resnet_152 |  384x288   |    **0.96**     |      1.65      |  **0.750**   |    0.743    |
| hrnet_w32  |  256x192   |    **0.54**     |      1.31      |  **0.746**   |    0.744    |
| hrnet_w32  |  384x288   |    **0.76**     |      2.00      |  **0.760**   |    0.758    |
| hrnet_w48  |  256x192   |    **0.66**     |      1.55      |  **0.756**   |    0.751    |
| hrnet_w48  |  384x288   |    **1.23**     |      2.20      |  **0.767**   |    0.763    |

## Hardware

- 8 NVIDIA Tesla V100 (32G) GPUs
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

## Software Environment

- Python 3.7
- PyTorch 1.4
- CUDA 10.1
- CUDNN 7.6.03
- NCCL 2.4.08
