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

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{zhang2020distribution,
  title={Distribution-aware coordinate representation for human pose estimation},
  author={Zhang, Feng and Zhu, Xiatian and Dai, Hanbin and Ye, Mao and Zhu, Ce},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7093--7102},
  year={2020}
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
| [pose_hrnet_w32_dark](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/mpii/hrnet_w32_mpii_256x256_dark.py) | 256x256 | 0.904 | 0.354 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256_dark-f1601c5b_20200927.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256_dark_20200927.log.json) |
| [pose_hrnet_w48_dark](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/mpii/hrnet_w48_mpii_256x256_dark.py) | 256x256 | 0.905 | 0.360 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_mpii_256x256_dark-0decd39f_20200927.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_mpii_256x256_dark_20200927.log.json) |
