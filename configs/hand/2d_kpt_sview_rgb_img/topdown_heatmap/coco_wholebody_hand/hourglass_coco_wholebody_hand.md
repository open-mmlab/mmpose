<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-319-46484-8_29">Hourglass (ECCV'2016)</a></summary>

```bibtex
@inproceedings{newell2016stacked,
  title={Stacked hourglass networks for human pose estimation},
  author={Newell, Alejandro and Yang, Kaiyu and Deng, Jia},
  booktitle={European conference on computer vision},
  pages={483--499},
  year={2016},
  organization={Springer}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Hand (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody-Hand val set

| Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hourglass_52](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hourglass52_coco_wholebody_hand_256x256.py) | 256x256 | 0.804 | 0.835 | 4.54 | [ckpt](https://download.openmmlab.com/mmpose/hand/hourglass/hourglass52_coco_wholebody_hand_256x256-7b05c6db_20210909.pth) | [log](https://download.openmmlab.com/mmpose/hand/hourglass/hourglass52_coco_wholebody_hand_256x256_20210909.log.json) |
