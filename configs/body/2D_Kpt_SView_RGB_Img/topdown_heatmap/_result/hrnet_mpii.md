
<!-- [ALGORITHM] -->

```bibtex
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
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
| [pose_hrnet_w32](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/mpii/hrnet_w32_mpii_256x256.py) | 256x256 | 0.900 | 0.334 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256-6c4f923f_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256_20200812.log.json) |
| [pose_hrnet_w48](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/mpii/hrnet_w48_mpii_256x256.py) | 256x256 | 0.901 | 0.337 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_mpii_256x256-92cab7bd_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_mpii_256x256_20200812.log.json) |
