# NPU (华为昇腾)

## 使用方法

首先，请参考[MMCV](https://mmcv.readthedocs.io/zh_CN/latest/get_started/build.html#npu-mmcv-full) 安装带有 NPU 支持的 MMCV。
使用如下命令，可以利用 4 个 NPU 训练模型（以 HRNet为例）：

```shell
bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py 4
```

或者，使用如下命令，在一个 NPU 上训练模型（以 HRNet为例）：

```shell
python tools/train.py configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py
```

## 经过验证的模型

|                         Model                         | Input Size |  AP   | AP50  | AP75  |  AR   | AR50  | Config                                                  | Download                                                  |
| :---------------------------------------------------: | :--------: | :---: | :---: | :---: | :---: | :---: | :------------------------------------------------------ | :-------------------------------------------------------- |
| [HigherHRNet](https://mmpose.readthedocs.io/zh_CN/0.x/papers/backbones.html#associative-embedding-higherhrnet-on-coco) |  512x512   | 0.670 | 0.859 | 0.732 | 0.724 | 0.893 | [config](https://github.com/open-mmlab/mmpose/blob/dev-0.x/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_512x512.py) | [log](https://download.openmmlab.com/mmpose/device/npu/hrnet_20230413_035450.log.json) |

**注意:**

- 如果没有特别标记，NPU 上的结果与使用 FP32 的 GPU 上的结果结果相同。

**以上所有模型权重及训练日志均由华为昇腾团队提供**
