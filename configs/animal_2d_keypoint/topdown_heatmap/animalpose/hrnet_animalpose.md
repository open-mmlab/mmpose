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
<summary align="right"><a href="http://openaccess.thecvf.com/content_ICCV_2019/html/Cao_Cross-Domain_Adaptation_for_Animal_Pose_Estimation_ICCV_2019_paper.html">Animal-Pose (ICCV'2019)</a></summary>

```bibtex
@InProceedings{Cao_2019_ICCV,
    author = {Cao, Jinkun and Tang, Hongyang and Fang, Hao-Shu and Shen, Xiaoyong and Lu, Cewu and Tai, Yu-Wing},
    title = {Cross-Domain Adaptation for Animal Pose Estimation},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

</details>

Results on AnimalPose validation set (1117 instances)

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_hrnet_w32](/configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py) |  256x256   | 0.740 |      0.959      |      0.833      | 0.780 |      0.965      | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256_20210426.log.json) |
| [pose_hrnet_w48](/configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w48_8xb64-210e_animalpose-256x256.py) |  256x256   | 0.738 |      0.958      |      0.831      | 0.778 |      0.962      | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_animalpose_256x256-34644726_20210426.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_animalpose_256x256_20210426.log.json) |
