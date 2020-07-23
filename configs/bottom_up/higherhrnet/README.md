# HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation

## Introduction
```
@inproceedings{cheng2020higherhrnet,
  title={HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation},
  author={Cheng, Bowen and Xiao, Bin and Wang, Jingdong and Shi, Honghui and Huang, Thomas S and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5386--5395},
  year={2020}
}
```

## Results and models

### Results on COCO val2017 without multi-scale test

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| HRNet-w32  | 512x512 | 0.677 | 0.870 | 0.738 | 0.723 | 0.890 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/bottom_up/higher_hrnet32_coco_512x512-8ae85183_20200713.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/bottom_up/higher_hrnet32_coco_512x512_20200713.log.json) |
| HRNet-w32  | 640x640 | 0.686 | 0.871 | 0.747 | 0.733 | 0.898 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/bottom_up/higher_hrnet32_coco_640x640-a22fe938_20200712.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/bottom_up/higher_hrnet32_coco_640x640_20200712.log.json) |
| HRNet-w48  | 512x512 | 0.686 | 0.873 | 0.741 | 0.731 | 0.892 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/bottom_up/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/bottom_up/higher_hrnet48_coco_512x512_20200712.log.json) |

### Results on COCO val2017 with multi-scale test

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| HRNet-w32  | 512x512 | 0.706 | 0.881 | 0.771 | 0.747 | 0.901 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/bottom_up/higher_hrnet32_coco_512x512-8ae85183_20200713.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/bottom_up/higher_hrnet32_coco_512x512_20200713.log.json) |
| HRNet-w32  | 640x640 | 0.706 | 0.880 | 0.770 | 0.749 | 0.902 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/bottom_up/higher_hrnet32_coco_640x640-a22fe938_20200712.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/bottom_up/higher_hrnet32_coco_640x640_20200712.log.json) |
| HRNet-w48  | 512x512 | 0.716 | 0.884 | 0.775 | 0.755 | 0.901 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/bottom_up/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/bottom_up/higher_hrnet48_coco_512x512_20200712.log.json) |
