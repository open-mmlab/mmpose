<!-- [BACKBONE] -->

```bibtex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
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
| [pose_mobilenet_v2](/configs/hand/2D_Kpt_SView_RGB_Img/top_down_heatmap/panoptic/mobilenetv2_panoptic_256x256.py) | 256x256 | 0.998 | 0.694 | 9.70 | [ckpt](https://download.openmmlab.com/mmpose/hand/mobilenetv2/mobilenetv2_panoptic_256x256-b733d98c_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/mobilenetv2/mobilenetv2_panoptic_256x256_20210330.log.json) |
