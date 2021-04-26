# HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{cheng2020higherhrnet,
  title={HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation},
  author={Cheng, Bowen and Xiao, Bin and Wang, Jingdong and Shi, Honghui and Huang, Thomas S and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5386--5395},
  year={2020}
}
```

## Results and models

### 2d Human Pose Estimation

#### Results on COCO val2017 without multi-scale test

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [HigherHRNet-w32](/configs/bottom_up/higherhrnet/coco/higher_hrnet32_coco_512x512.py)  | 512x512 | 0.677 | 0.870 | 0.738 | 0.723 | 0.890 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_512x512-8ae85183_20200713.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_512x512_20200713.log.json) |
| [HigherHRNet-w32](/configs/bottom_up/higherhrnet/coco/higher_hrnet32_coco_640x640.py)  | 640x640 | 0.686 | 0.871 | 0.747 | 0.733 | 0.898 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_640x640-a22fe938_20200712.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_640x640_20200712.log.json) |
| [HigherHRNet-w48](/configs/bottom_up/higherhrnet/coco/higher_hrnet48_coco_512x512.py)  | 512x512 | 0.686 | 0.873 | 0.741 | 0.731 | 0.892 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512_20200712.log.json) |

#### Results on COCO val2017 with multi-scale test. 3 default scales (\[2, 1, 0.5\]) are used

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [HigherHRNet-w32](/configs/bottom_up/higherhrnet/coco/higher_hrnet32_coco_512x512.py)  | 512x512 | 0.706 | 0.881 | 0.771 | 0.747 | 0.901 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_512x512-8ae85183_20200713.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_512x512_20200713.log.json) |
| [HigherHRNet-w32](/configs/bottom_up/higherhrnet/coco/higher_hrnet32_coco_640x640.py)  | 640x640 | 0.706 | 0.880 | 0.770 | 0.749 | 0.902 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_640x640-a22fe938_20200712.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_640x640_20200712.log.json) |
| [HigherHRNet-w48](/configs/bottom_up/higherhrnet/coco/higher_hrnet48_coco_512x512.py)  | 512x512 | 0.716 | 0.884 | 0.775 | 0.755 | 0.901 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512_20200712.log.json) |

#### Results on CrowdPose test without multi-scale test

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP (E) | AP (M) | AP (H) | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: | :------: |
| [HigherHRNet-w32](/configs/bottom_up/higherhrnet/crowdpose/higher_hrnet32_crowdpose_512x512.py)  | 512x512 | 0.655 | 0.859 | 0.705 | 0.728 | 0.660 | 0.577 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_crowdpose_512x512-1aa4a132_20201017.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_crowdpose_512x512_20201017.log.json) |

#### Results on CrowdPose test with multi-scale test. 2 scales (\[2, 1\]) are used

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP (E) | AP (M) | AP (H) | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: | :------: |
| [HigherHRNet-w32](/configs/bottom_up/higherhrnet/crowdpose/higher_hrnet32_crowdpose_512x512.py)  | 512x512 | 0.661 | 0.864 | 0.710 | 0.742 | 0.670 | 0.566 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_crowdpose_512x512-1aa4a132_20201017.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_crowdpose_512x512_20201017.log.json) |

#### Results on AIC validation set without multi-scale test

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [HigherHRNet-w32](/configs/bottom_up/higherhrnet/aic/higher_hrnet32_aic_512x512.py)  | 512x512 | 0.315 | 0.710 | 0.243 | 0.379 | 0.757 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_aic_512x512-9a674c33_20210130.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_aic_512x512_20210130.log.json) |

#### Results on AIC validation set with multi-scale test. 3 default scales (\[2, 1, 0.5\]) are used

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [HigherHRNet-w32](/configs/bottom_up/higherhrnet/aic/higher_hrnet32_aic_512x512.py)  | 512x512 | 0.323 | 0.718 | 0.254 | 0.379 | 0.758 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_aic_512x512-9a674c33_20210130.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_aic_512x512_20210130.log.json) |
