<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html">SimpleBaseline2D (ECCV'2018)</a></summary>

```bibtex
@inproceedings{xiao2018simple,
  title={Simple baselines for human pose estimation and tracking},
  author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={466--481},
  year={2018}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1906.05586">ATRW (ACM MM'2020)</a></summary>

```bibtex
@inproceedings{li2020atrw,
  title={ATRW: A Benchmark for Amur Tiger Re-identification in the Wild},
  author={Li, Shuyuan and Li, Jianguo and Tang, Hanlin and Qian, Rui and Lin, Weiyao},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2590--2598},
  year={2020}
}
```

</details>

Results on ATRW validation set

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_resnet_50](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/atrw/res50_atrw_256x256.py) |  256x256   | 0.900 |      0.973      |      0.932      | 0.929 |      0.985      | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_atrw_256x256-546c4594_20210414.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_atrw_256x256_20210414.log.json) |
| [pose_resnet_101](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/atrw/res101_atrw_256x256.py) |  256x256   | 0.898 |      0.973      |      0.936      | 0.927 |      0.985      | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_atrw_256x256-da93f371_20210414.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_atrw_256x256_20210414.log.json) |
| [pose_resnet_152](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/atrw/res152_atrw_256x256.py) |  256x256   | 0.896 |      0.973      |      0.931      | 0.927 |      0.985      | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_atrw_256x256-2bb8e162_20210414.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_atrw_256x256_20210414.log.json) |
