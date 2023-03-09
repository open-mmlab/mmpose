<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1906.04016">PoseWarper (NeurIPS'2019)</a></summary>

```bibtex
@inproceedings{NIPS2019_gberta,
title = {Learning Temporal Pose Estimation from Sparsely Labeled Videos},
author = {Bertasius, Gedas and Feichtenhofer, Christoph, and Tran, Du and Shi, Jianbo, and Torresani, Lorenzo},
booktitle = {Advances in Neural Information Processing Systems 33},
year = {2019},
}
```

</details>

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
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Andriluka_PoseTrack_A_Benchmark_CVPR_2018_paper.html">PoseTrack18 (CVPR'2018)</a></summary>

```bibtex
@inproceedings{andriluka2018posetrack,
  title={Posetrack: A benchmark for human pose estimation and tracking},
  author={Andriluka, Mykhaylo and Iqbal, Umar and Insafutdinov, Eldar and Pishchulin, Leonid and Milan, Anton and Gall, Juergen and Schiele, Bernt},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5167--5176},
  year={2018}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48">COCO (ECCV'2014)</a></summary>

```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

</details>

Note that the training of PoseWarper can be split into two stages.

The first-stage is trained with the pre-trained [checkpoint](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288-314c8528_20200708.pth) on COCO dataset, and the main backbone is fine-tuned on PoseTrack18 in a single-frame setting.

The second-stage is trained with the last [checkpoint](https://download.openmmlab.com/mmpose/top_down/posewarper/hrnet_w48_posetrack18_384x288_posewarper_stage1-08b632aa_20211130.pth) from the first stage, and the warping offsets are learned in a multi-frame setting while the backbone is frozen.

Results on PoseTrack2018 val with ground-truth bounding boxes

| Arch                                                 | Input Size | Head | Shou | Elb  | Wri  | Hip  | Knee | Ankl | Total |                         ckpt                          |                         log                          |
| :--------------------------------------------------- | :--------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :---: | :---------------------------------------------------: | :--------------------------------------------------: |
| [pose_hrnet_w48](/configs/body/2d_kpt_sview_rgb_vid/posewarper/posetrack18/hrnet_w48_posetrack18_384x288_posewarper_stage2.py) |  384x288   | 88.2 | 90.3 | 86.1 | 81.6 | 81.8 | 83.8 | 81.5 | 85.0  | [ckpt](https://download.openmmlab.com/mmpose/top_down/posewarper/hrnet_w48_posetrack18_384x288_posewarper_stage2-4abf88db_20211130.pth) | [log](https://download.openmmlab.com/mmpose/top_down/posewarper/hrnet_w48_posetrack18_384x288_posewarper_stage2_20211130.log.json) |

Results on PoseTrack2018 val with precomputed human bounding boxes from PoseWarper supplementary data files from [this link](https://www.dropbox.com/s/ygfy6r8nitoggfq/PoseWarper_supp_files.zip?dl=0)<sup>1</sup>.

| Arch                                                 | Input Size | Head | Shou | Elb  | Wri  | Hip  | Knee | Ankl | Total |                         ckpt                          |                         log                          |
| :--------------------------------------------------- | :--------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :---: | :---------------------------------------------------: | :--------------------------------------------------: |
| [pose_hrnet_w48](/configs/body/2d_kpt_sview_rgb_vid/posewarper/posetrack18/hrnet_w48_posetrack18_384x288_posewarper_stage2.py) |  384x288   | 81.8 | 85.6 | 82.7 | 77.2 | 76.8 | 79.0 | 74.4 | 79.8  | [ckpt](https://download.openmmlab.com/mmpose/top_down/posewarper/hrnet_w48_posetrack18_384x288_posewarper_stage2-4abf88db_20211130.pth) | [log](https://download.openmmlab.com/mmpose/top_down/posewarper/hrnet_w48_posetrack18_384x288_posewarper_stage2_20211130.log.json) |

<sup>1</sup> Please download the precomputed human bounding boxes on PoseTrack2018 val from `$PoseWarper_supp_files/posetrack18_precomputed_boxes/val_boxes.json` and place it here: `$mmpose/data/posetrack18/posetrack18_precomputed_boxes/val_boxes.json` to be consistent with the [config](/configs/body/2d_kpt_sview_rgb_vid/posewarper/posetrack18/hrnet_w48_posetrack18_384x288_posewarper_stage2.py). Please refer to [DATA Preparation](/docs/en/tasks/2d_body_keypoint.md) for more detail about data preparation.
