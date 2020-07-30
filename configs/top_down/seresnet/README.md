# Squeeze-and-excitation networks

## Introduction
```
@inproceedings{hu2018squeeze,
  title={Squeeze-and-excitation networks},
  author={Hu, Jie and Shen, Li and Sun, Gang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={7132--7141},
  year={2018}
}
```

## Results and models

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| pose_seresnet_50  | 256x192 | 0.728 | 0.900 | 0.809 | 0.784 | 0.940 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/seresnet/seresnet50_coco_256x192-25058b66_20200727.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/seresnet/seresnet50_coco_256x192_20200727.log.json) |
| pose_seresnet_50  | 384x288 | 0.748 | 0.905 | 0.819 | 0.799 | 0.941 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/seresnet/seresnet50_coco_384x288-bc0b7680_20200727.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/seresnet/seresnet50_coco_384x288_20200727.log.json) |
| pose_seresnet_101 | 256x192 | 0.734 | 0.904 | 0.815 | 0.790 | 0.942 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/seresnet/seresnet101_coco_256x192-83f29c4d_20200727.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/seresnet/seresnet101_coco_256x192_20200727.log.json) |
| pose_seresnet_101 | 384x288 | 0.753 | 0.907 | 0.823 | 0.805 | 0.943 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/seresnet/seresnet101_coco_384x288-48de1709_20200727.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/seresnet/seresnet101_coco_384x288_20200727.log.json) |
| pose_seresnet_152* | 256x192 | 0.730 | 0.899 | 0.810 | 0.786 | 0.940 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/seresnet/seresnet152_coco_256x192-1c628d79_20200727.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/seresnet/seresnet152_coco_256x192_20200727.log.json) |
| pose_seresnet_152* | 384x288 | 0.753 | 0.906 | 0.823 | 0.806 | 0.945 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/seresnet/seresnet152_coco_384x288-58b23ee8_20200727.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/seresnet/seresnet152_coco_384x288_20200727.log.json) |

Note that * means without imagenet pre-training.
