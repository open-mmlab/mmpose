# Model Zoo

## Result

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| pose_resnet_50  | 256x192 | 0.718 | 0.898 | 0.795 | 0.773 | 0.937 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/resnet/res50_coco_256x192_20200709.log.json) |
| pose_resnet_50  | 384x288 | 0.731 | 0.900 | 0.799 | 0.783 | 0.931 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/resnet/res50_coco_384x288-e6f795e9_20200709.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/resnet/res50_coco_384x288_20200709.log.json) |
| pose_resnet_101 | 256x192 | 0.726 | 0.899 | 0.806 | 0.781 | 0.939 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/resnet/res101_coco_256x192-6e6babf0_20200708.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/resnet/res101_coco_256x192_20200708.log.json) |
| pose_resnet_101 | 384x288 | 0.748 | 0.905 | 0.817 | 0.798 | 0.940 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/resnet/res101_coco_384x288-8c71bdc9_20200709.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/resnet/res101_coco_384x288_20200709.log.json) |
| pose_resnet_152 | 256x192 | 0.735 | 0.905 | 0.812 | 0.790 | 0.943 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/resnet/res152_coco_256x192-f6e307c2_20200709.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/resnet/res152_coco_256x192_20200709.log.json) |
| pose_resnet_152 | 384x288 | 0.750 | 0.908 | 0.821 | 0.800 | 0.942 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/resnet/res152_coco_384x288-3860d4c9_20200709.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/resnet/res152_coco_384x288_20200709.log.json) |
| pose_hrnet_w32  | 256x192 | 0.746 | 0.904 | 0.819 | 0.799 | 0.942 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192_20200708.log.json) |
| pose_hrnet_w32  | 384x288 | 0.760 | 0.906 | 0.829 | 0.810 | 0.943 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w32_coco_384x288-d9f0d786_20200708.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w32_coco_384x288_20200708.log.json) |
| pose_hrnet_w48  | 256x192 | 0.756 | 0.907 | 0.825 | 0.806 | 0.942 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192_20200708.log.json) |
| pose_hrnet_w48  | 384x288 | 0.767 | 0.910 | 0.831 | 0.816 | 0.946 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288-314c8528_20200708.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_20200708.log.json) |
| pose_scnet_50   | 256x192 | 0.728 | 0.899 | 0.807 | 0.784 | 0.938 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/scnet/scnet50_coco_256x192-6920f829_20200709.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/scnet/scnet50_coco_256x192_20200709.log.json) |
| pose_scnet_50  | 384x288 | 0.751 | 0.906 | 0.818 | 0.802 | 0.943 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/scnet/scnet50_coco_384x288-9cacd0ea_20200709.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/scnet/scnet50_coco_384x288_20200709.log.json) |
| pose_scnet_101  | 256x192 | 0.733 | 0.903 | 0.813 | 0.790 | 0.941 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/scnet/scnet101_coco_256x192-6d348ef9_20200709.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/scnet/scnet101_coco_256x192_20200709.log.json) |
| pose_scnet_101  | 384x288 | 0.752 | 0.906 | 0.823 | 0.804 | 0.943 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/scnet/scnet101_coco_384x288-0b6e631b_20200709.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/scnet/scnet101_coco_384x288_20200709.log.json) |
| dark_pose_resnet_50 | 256x192 | 0.724 | 0.898 | 0.800 | 0.777 | 0.936 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/resnet/res50_coco_256x192_dark-43379d20_20200709.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/resnet/res50_coco_256x192_dark_20200709.log.json) |
| pose_hourglass_52 | 256x256 | 0.726 | 0.896 | 0.799 | 0.780 | 0.934 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/hourglass/hourglass52_coco_256x256-4ec713ba_20200709.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/hourglass/hourglass52_coco_256x256_20200709.log.json) |


### Pretrained backbones on ImageNet

| Arch |  ckpt |
| :----------------- | :-----------: |
| resnet50-19c8e357.pth | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/pretrain_models/resnet50-19c8e357.pth)
| resnet101-5d3b4d8f.pth | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/pretrain_models/resnet101-5d3b4d8f.pth)
| resnet152-b121ed2d.pth | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/pretrain_models/resnet152-b121ed2d.pth)
| hrnet_w32-36af842e.pth  | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth) |
| hrnet_w48-8ef0771d.pth | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/pretrain_models/hrnet_w48-8ef0771d.pth)
| scnet50-7ef0a199.pth | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/pretrain_models/scnet50-7ef0a199.pth)
| scnet101-94250a77.pth | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/pretrain_models/scnet101-94250a77.pth)
