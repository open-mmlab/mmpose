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
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Face (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody-Face val set

| Arch                                                          | Input Size |  NME   |                              ckpt                              |                              log                              |
| :------------------------------------------------------------ | :--------: | :----: | :------------------------------------------------------------: | :-----------------------------------------------------------: |
| [pose_scnet_50](/configs/face_2d_keypoint/topdown_heatmap/coco_wholebody_face/td-hm_scnet50_8xb32-60e_coco-wholebody-face-256x256.py) |  256x256   | 0.0567 | [ckpt](https://download.openmmlab.com/mmpose/face/scnet/scnet50_coco_wholebody_face_256x256-a0183f5f_20210909.pth) | [log](https://download.openmmlab.com/mmpose/face/scnet/scnet50_coco_wholebody_face_256x256_20210909.log.json) |
