# RTMPose Benchmarks

简体中文 | [English](./README.md)

欢迎社区用户在不同硬件设备上进行推理速度测试，贡献到本项目目录下。

当前已测试：

- CPU
  - Intel i7-11700
- GPU
  - NVIDIA GeForce 1660 Ti
- ARM
  - Snapdragon 865

### 人体 2d 关键点 (17 Keypoints)

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

|                                Config                                 | Input Size | ORT-Latency(ms)<sup><br>(i7-11700) | TRT-FP16-Latency(ms)<sup><br>(GTX 1660Ti) | ncnn-FP16-Latency(ms)<sup><br>(Snapdragon 865) |
| :-------------------------------------------------------------------: | :--------: | :--------------------------------: | :---------------------------------------: | :--------------------------------------------: |
| [RTMPose-t](../rtmpose/body_2d_keypoint/rtmpose-tiny_8xb256-420e_coco-256x192.py) |  256x192   |                3.20                |                   1.06                    |                      9.02                      |
| [RTMPose-s](../rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py) |  256x192   |                4.48                |                   1.39                    |                     13.89                      |
| [RTMPose-m](../rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py) |  256x192   |               11.06                |                   2.29                    |                     26.44                      |
| [RTMPose-l](../rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-256x192.py) |  256x192   |               18.85                |                   3.46                    |                     45.37                      |
| [RTMPose-m](../rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-384x288.py) |  384x288   |               24.78                |                   3.66                    |                     26.44                      |
| [RTMPose-l](../rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-384x288.py) |  384x288   |                 -                  |                   6.05                    |                       -                        |

### 人体全身 2d 关键点 (133 Keypoints)

### Model Info

| Config                                                                                       | Input Size | Whole AP | Whole AR | FLOPS(G) |
| :------------------------------------------------------------------------------------------- | :--------: | :------: | :------: | :------: |
| [RTMPose-m](../rtmpose/wholebody_2d_keypoint/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py) |  256x192   |   60.4   |   66.7   |   2.22   |
| [RTMPose-l](../rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py) |  256x192   |   63.2   |   69.4   |   4.52   |
| [RTMPose-l](../rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py) |  384x288   |   67.0   |   72.3   |  10.07   |

### Speed Benchmark

| Config                                                                       | Input Size | ORT-Latency(ms)<sup><br>(i7-11700) | TRT-FP16-Latency(ms)<sup><br>(GTX 1660Ti) | TRT-FP16-Latency(ms)<sup><br>(GTX 3090) |
| :--------------------------------------------------------------------------- | :--------: | :--------------------------------: | :---------------------------------------: | :-------------------------------------: |
| [RTMPose-m](../rtmpose/wholebody_2d_keypoint/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py) |  256x192   |               13.50                |                   4.00                    |                  1.84                   |
| [RTMPose-l](../rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py) |  256x192   |               23.41                |                   5.67                    |                  2.61                   |
| [RTMPose-l](../rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py) |  384x288   |               44.58                |                   7.68                    |                  4.24                   |

## 如何测试推理速度

我们使用 MMDeploy 提供的 `tools/profiler.py` 脚本进行模型测速。

用户需要准备一个存放测试图片的文件夹`./test_images`，profiler 将随机从该目录下抽取图片用于模型测速。

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

If you want to learn more details of profiler, you can refer to the [Profiler Docs](https://mmdeploy.readthedocs.io/en/1.x/02-how-to-run/useful_tools.html#profiler).
