# NPU (HUAWEI Ascend)

## Usage

Please refer to the [building documentation of MMCV](https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-mmcv-full-on-ascend-npu-machine) to install MMCV on NPU devices.

Here we use 4 NPUs on your computer to train the model with the following command:

```shell
bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py 4
```

Also, you can use only one NPU to train the model with the following command:

```shell
python tools/train.py configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py
```

## Models Results

|   Model   | Input Size |  	AP  | 	AP50  | 	AP75  |  	AR   | 	AR50  | Config                                                                                                                           | Download                                                                                                                   |
|:---------:|:----------:|:-----:|:------:|:------:|:------:|:------:|:---------------------------------------------------------------------------------------------------------------------------------| :------------------------------------------------------------------------------------------------------------------------- |
| [HigherHRNet](https://mmpose.readthedocs.io/zh_CN/0.x/papers/backbones.html#associative-embedding-higherhrnet-on-coco) |  512x512   | 0.670 | 	0.859 | 	0.732 | 	0.724 | 	0.893 | [config](https://github.com/open-mmlab/mmpose/blob/dev-0.x/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_512x512.py) | [log](https://download.openmmlab.com/mmpose/device/npu/hrnet_20230413_035450.log.json)     |


**Notes:**

- If not specially marked, the results on NPU with amp are the basically same as those on the GPU with FP32.

**All above models are provided by Huawei Ascend group.**
