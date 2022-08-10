<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html">HRNet (CVPR'2019)</a></summary>

```bibtex
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://www.ncbi.nlm.nih.gov/pmc/articles/pmc7874091/">MacaquePose (bioRxiv'2020)</a></summary>

```bibtex
@article{labuguen2020macaquepose,
  title={MacaquePose: A novel ‘in the wild’macaque monkey pose dataset for markerless motion capture},
  author={Labuguen, Rollyn and Matsumoto, Jumpei and Negrete, Salvador and Nishimaru, Hiroshi and Nishijo, Hisao and Takada, Masahiko and Go, Yasuhiro and Inoue, Ken-ichi and Shibata, Tomohiro},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
```

</details>

Results on MacaquePose with ground-truth detection bounding boxes

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_hrnet_w32](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/macaque/hrnet_w32_macaque_256x192.py) |  256x192   | 0.814 |      0.953      |      0.918      | 0.851 |      0.969      | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_macaque_256x192-f7e9e04f_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_macaque_256x192_20210407.log.json) |
| [pose_hrnet_w48](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/macaque/hrnet_w48_macaque_256x192.py) |  256x192   | 0.818 |      0.963      |      0.917      | 0.855 |      0.971      | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_macaque_256x192-9b34b02a_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_macaque_256x192_20210407.log.json) |
