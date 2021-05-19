<!-- [BACKBONE] -->

```bibtex
@inproceedings{hu2018squeeze,
  title={Squeeze-and-excitation networks},
  author={Hu, Jie and Shen, Li and Sun, Gang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={7132--7141},
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
| [pose_seresnet_50](/configs/body/2D_Kpt_SView_RGB_Img/top_down_heatmap/mpii/seresnet50_mpii_256x256.py) | 256x256 | 0.884 | 0.292 | [ckpt](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet50_mpii_256x256-1bb21f79_20200927.pth) | [log](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet50_mpii_256x256_20200927.log.json) |
| [pose_seresnet_101](/configs/body/2D_Kpt_SView_RGB_Img/top_down_heatmap/mpii/seresnet101_mpii_256x256.py) | 256x256 | 0.884 | 0.295 | [ckpt](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet101_mpii_256x256-0ba14ff5_20200927.pth) | [log](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet101_mpii_256x256_20200927.log.json) |
| [pose_seresnet_152\*](/configs/body/2D_Kpt_SView_RGB_Img/top_down_heatmap/mpii/seresnet152_mpii_256x256.py) | 256x256 | 0.884 | 0.287 | [ckpt](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet152_mpii_256x256-6ea1e774_20200927.pth) | [log](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet152_mpii_256x256_20200927.log.json) |

Note that \* means without imagenet pre-training.
