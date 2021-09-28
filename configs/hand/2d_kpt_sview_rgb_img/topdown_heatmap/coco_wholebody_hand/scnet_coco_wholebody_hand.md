<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Improving_Convolutional_Networks_With_Self-Calibrated_Convolutions_CVPR_2020_paper.html">SCNet (CVPR'2020)</a></summary>

```bibtex
@inproceedings{liu2020improving,
  title={Improving Convolutional Networks with Self-Calibrated Convolutions},
  author={Liu, Jiang-Jiang and Hou, Qibin and Cheng, Ming-Ming and Wang, Changhu and Feng, Jiashi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10096--10105},
  year={2020}
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
| :--------: | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_scnet_50](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/scnet50_coco_wholebody_hand_256x256.py) | 256x256 | 0.803 | 0.834 | 4.55 | [ckpt](https://download.openmmlab.com/mmpose/hand/scnet/scnet50_coco_wholebody_hand_256x256-e73414c7_20210909.pth) | [log](https://download.openmmlab.com/mmpose/hand/scnet/scnet50_coco_wholebody_hand_256x256_20210909.log.json) |
