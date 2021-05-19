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
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt}
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

#### Results on MPII val set

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/body/2D_Kpt_SView_RGB_Img/top_down_heatmap/mpii/res50_mpii_256x256.py) | 256x256 | 0.882 | 0.286 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_256x256-418ffc88_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_256x256_20200812.log.json) |
| [pose_resnet_101](/configs/body/2D_Kpt_SView_RGB_Img/top_down_heatmap/mpii/res101_mpii_256x256.py) | 256x256 | 0.888 | 0.290 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_mpii_256x256-416f5d71_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_mpii_256x256_20200812.log.json) |
| [pose_resnet_152](/configs/body/2D_Kpt_SView_RGB_Img/top_down_heatmap/mpii/res152_mpii_256x256.py) | 256x256 | 0.889 | 0.303 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_mpii_256x256-3ecba29d_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_mpii_256x256_20200812.log.json) |
