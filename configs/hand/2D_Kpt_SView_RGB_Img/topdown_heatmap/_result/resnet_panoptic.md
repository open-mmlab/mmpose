<!-- [ALGORITHM] -->

```bibtex
@inproceedings{xiao2018simple,
  title={Simple baselines for human pose estimation and tracking},
  author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={466--481},
  year={2018}
}
```

<!-- [DATASET] -->

```bibtex
@inproceedings{simon2017hand,
  title={Hand keypoint detection in single images using multiview bootstrapping},
  author={Simon, Tomas and Joo, Hanbyul and Matthews, Iain and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={1145--1153},
  year={2017}
}
```

#### Results on CMU Panoptic (MPII+NZSL val set)

| Arch  | Input Size | PCKh@0.7 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/hand/2D_Kpt_SView_RGB_Img/topdown_heatmap/panoptic/res50_panoptic_256x256.py) | 256x256 | 0.999 | 0.713 | 9.00 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_panoptic_256x256-4eafc561_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_panoptic_256x256_20210330.log.json) |
