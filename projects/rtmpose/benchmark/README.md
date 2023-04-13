# RTMPose Benchmarks

English | [简体中文](./README_CN.md)

Community users are welcome to contribute to this project directory by performing inference speed tests on different hardware devices.

Currently tested:

- CPU
  - Intel i7-11700
- GPU
  - NVIDIA GeForce 1660 Ti
  - NVIDIA GeForce RTX 3090
- Nvidia Jetson
  - AGX Orin
  - Orin NX
- ARM
  - Snapdragon 865

## Body 2d (17 Keypoints)

### Model Info

|                                      Config                                       | Input Size | AP<sup><br>(COCO) | Params(M) | FLOPS(G) |
| :-------------------------------------------------------------------------------: | :--------: | :---------------: | :-------: | :------: |
| [RTMPose-t](../rtmpose/body_2d_keypoint/rtmpose-tiny_8xb256-420e_coco-256x192.py) |  256x192   |       68.5        |   3.34    |   0.36   |
|  [RTMPose-s](../rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py)   |  256x192   |       72.2        |   5.47    |   0.68   |
|  [RTMPose-m](../rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py)   |  256x192   |       75.8        |   13.59   |   1.93   |
|  [RTMPose-l](../rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-256x192.py)   |  256x192   |       76.5        |   27.66   |   4.16   |
|  [RTMPose-m](../rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-384x288.py)   |  384x288   |       77.0        |   13.72   |   4.33   |
|  [RTMPose-l](../rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-384x288.py)   |  384x288   |       77.3        |   27.79   |   9.35   |

### Speed Benchmark

- Numbers displayed in the table are inference latencies in millisecond(ms).

|   Config    | Input Size | ORT<sup><br>(i7-11700) | TRT-FP16<sup><br>(GTX 1660Ti) | TRT-FP16<sup><br>(RTX 3090) | ncnn-FP16<sup><br>(Snapdragon 865) | TRT-FP16<sup><br>(Jetson AGX Orin) | TRT-FP16<sup><br>(Jetson Orin NX) |
| :---------: | :--------: | :--------------------: | :---------------------------: | :-------------------------: | :--------------------------------: | :--------------------------------: | :-------------------------------: |
| [RTMPose-t](../rtmpose/body_2d_keypoint/rtmpose-tiny_8xb256-420e_coco-256x192.py) |  256x192   |          3.20          |             1.06              |            0.98             |                9.02                |                1.63                |               1.97                |
| [RTMPose-s](../rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py) |  256x192   |          4.48          |             1.39              |            1.12             |               13.89                |                1.85                |               2.18                |
| [RTMPose-m](../rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py) |  256x192   |         11.06          |             2.29              |            1.18             |               26.44                |                2.72                |               3.35                |
| [RTMPose-l](../rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-256x192.py) |  256x192   |         18.85          |             3.46              |            1.37             |               45.37                |                3.67                |               4.78                |
| [RTMPose-m](../rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-384x288.py) |  384x288   |         24.78          |             3.66              |            1.20             |               26.44                |                3.45                |               5.08                |
| [RTMPose-l](../rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-384x288.py) |  384x288   |           -            |             6.05              |            1.74             |                 -                  |                4.93                |               7.23                |

## WholeBody 2d (133 Keypoints)

### Model Info

| Config                                                                                       | Input Size | Whole AP | Whole AR | FLOPS(G) |
| :------------------------------------------------------------------------------------------- | :--------: | :------: | :------: | :------: |
| [RTMPose-m](../rtmpose/wholebody_2d_keypoint/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py) |  256x192   |   60.4   |   66.7   |   2.22   |
| [RTMPose-l](../rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py) |  256x192   |   63.2   |   69.4   |   4.52   |
| [RTMPose-l](../rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py) |  384x288   |   67.0   |   72.3   |  10.07   |

### Speed Benchmark

- Numbers displayed in the table are inference latencies in millisecond(ms).
- Data from different community users are separated by `|`.

| Config                                        | Input Size | ORT<sup><br>(i7-11700) | TRT-FP16<sup><br>(GTX 1660Ti) | TRT-FP16<sup><br>(RTX 3090) | TRT-FP16<sup><br>(Jetson AGX Orin) | TRT-FP16<sup><br>(Jetson Orin NX) |
| :-------------------------------------------- | :--------: | :--------------------: | :---------------------------: | :-------------------------: | :--------------------------------: | :-------------------------------: |
| [RTMPose-m](../rtmpose/wholebody_2d_keypoint/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py) |  256x192   |         13.50          |             4.00              |        1.17 \| 1.84         |                2.79                |               3.51                |
| [RTMPose-l](../rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py) |  256x192   |         23.41          |             5.67              |        1.44 \| 2.61         |                3.80                |               4.95                |
| [RTMPose-l](../rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py) |  384x288   |         44.58          |             7.68              |        1.75 \| 4.24         |                5.08                |               7.20                |

## How To Test Speed

If you need to test the inference speed of the model under the deployment framework, MMDeploy provides a convenient `tools/profiler.py` script.

The user needs to prepare a folder for the test images `./test_images`, the profiler will randomly read images from this directory for the model speed test.

```shell
python tools/profiler.py \
    configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
    {RTMPOSE_PROJECT}/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    ../test_images \
    --model {WORK_DIR}/end2end.onnx \
    --shape 256x192 \
    --device cpu \
    --warmup 50 \
    --num-iter 200
```

The result is as follows:

```shell
01/30 15:06:35 - mmengine - INFO - [onnxruntime]-70 times per count: 8.73 ms, 114.50 FPS
01/30 15:06:36 - mmengine - INFO - [onnxruntime]-90 times per count: 9.05 ms, 110.48 FPS
01/30 15:06:37 - mmengine - INFO - [onnxruntime]-110 times per count: 9.87 ms, 101.32 FPS
01/30 15:06:37 - mmengine - INFO - [onnxruntime]-130 times per count: 9.99 ms, 100.10 FPS
01/30 15:06:38 - mmengine - INFO - [onnxruntime]-150 times per count: 10.39 ms, 96.29 FPS
01/30 15:06:39 - mmengine - INFO - [onnxruntime]-170 times per count: 10.77 ms, 92.86 FPS
01/30 15:06:40 - mmengine - INFO - [onnxruntime]-190 times per count: 10.98 ms, 91.05 FPS
01/30 15:06:40 - mmengine - INFO - [onnxruntime]-210 times per count: 11.19 ms, 89.33 FPS
01/30 15:06:41 - mmengine - INFO - [onnxruntime]-230 times per count: 11.16 ms, 89.58 FPS
01/30 15:06:42 - mmengine - INFO - [onnxruntime]-250 times per count: 11.06 ms, 90.41 FPS
----- Settings:
+------------+---------+
| batch size |    1    |
|   shape    | 256x192 |
| iterations |   200   |
|   warmup   |    50   |
+------------+---------+
----- Results:
+--------+------------+---------+
| Stats  | Latency/ms |   FPS   |
+--------+------------+---------+
|  Mean  |   11.060   |  90.412 |
| Median |   11.852   |  84.375 |
|  Min   |   7.812    | 128.007 |
|  Max   |   13.690   |  73.044 |
+--------+------------+---------+
```

If you want to learn more details of profiler, you can refer to the [Profiler Docs](https://mmdeploy.readthedocs.io/en/latest/02-how-to-run/useful_tools.html#profiler).
