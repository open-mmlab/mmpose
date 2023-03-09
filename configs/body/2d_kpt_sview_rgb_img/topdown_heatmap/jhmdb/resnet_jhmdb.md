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

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html">ResNet (CVPR'2016)</a></summary>

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://www.cv-foundation.org/openaccess/content_iccv_2013/html/Jhuang_Towards_Understanding_Action_2013_ICCV_paper.html">JHMDB (ICCV'2013)</a></summary>

```bibtex
@inproceedings{Jhuang:ICCV:2013,
  title = {Towards understanding action recognition},
  author = {H. Jhuang and J. Gall and S. Zuffi and C. Schmid and M. J. Black},
  booktitle = {International Conf. on Computer Vision (ICCV)},
  month = Dec,
  pages = {3192-3199},
  year = {2013}
}
```

</details>

Results on Sub-JHMDB dataset

The models are pre-trained on MPII dataset only. *NO* test-time augmentation (multi-scale /rotation testing) is used.

- Normalized by Person Size

| Split   |                        Arch                        | Input Size | Head | Sho  | Elb  | Wri  | Hip  | Knee | Ank  | Mean |                        ckpt                         |                        log                         |
| :------ | :------------------------------------------------: | :--------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :-------------------------------------------------: | :------------------------------------------------: |
| Sub1    | [pose_resnet_50](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/res50_jhmdb_sub1_256x256.py) |  256x256   | 99.1 | 98.0 | 93.8 | 91.3 | 99.4 | 96.5 | 92.8 | 96.1 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub1_256x256-932cb3b4_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub1_256x256_20201122.log.json) |
| Sub2    | [pose_resnet_50](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/res50_jhmdb_sub2_256x256.py) |  256x256   | 99.3 | 97.1 | 90.6 | 87.0 | 98.9 | 96.3 | 94.1 | 95.0 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub2_256x256-83d606f7_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub2_256x256_20201122.log.json) |
| Sub3    | [pose_resnet_50](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/res50_jhmdb_sub3_256x256.py) |  256x256   | 99.0 | 97.9 | 94.0 | 91.6 | 99.7 | 98.0 | 94.7 | 96.7 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub3_256x256-c4ec1a0b_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub3_256x256_20201122.log.json) |
| Average |                   pose_resnet_50                   |  256x256   | 99.2 | 97.7 | 92.8 | 90.0 | 99.3 | 96.9 | 93.9 | 96.0 |                          -                          |                         -                          |
| Sub1    | [pose_resnet_50 (2 Deconv.)](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/res50_2deconv_jhmdb_sub1_256x256.py) |  256x256   | 99.1 | 98.5 | 94.6 | 92.0 | 99.4 | 94.6 | 92.5 | 96.1 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub1_256x256-f0574a52_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub1_256x256_20201122.log.json) |
| Sub2    | [pose_resnet_50 (2 Deconv.)](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/res50_2deconv_jhmdb_sub2_256x256.py) |  256x256   | 99.3 | 97.8 | 91.0 | 87.0 | 99.1 | 96.5 | 93.8 | 95.2 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub2_256x256-f63af0ff_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub2_256x256_20201122.log.json) |
| Sub3    | [pose_resnet_50 (2 Deconv.)](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/res50_2deconv_jhmdb_sub3_256x256.py) |  256x256   | 98.8 | 98.4 | 94.3 | 92.1 | 99.8 | 97.5 | 93.8 | 96.7 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub3_256x256-c4bc2ddb_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub3_256x256_20201122.log.json) |
| Average |             pose_resnet_50 (2 Deconv.)             |  256x256   | 99.1 | 98.2 | 93.3 | 90.4 | 99.4 | 96.2 | 93.4 | 96.0 |                          -                          |                         -                          |

- Normalized by Torso Size

| Split   |                        Arch                        | Input Size | Head | Sho  | Elb  | Wri  | Hip  | Knee | Ank  | Mean |                        ckpt                         |                        log                         |
| :------ | :------------------------------------------------: | :--------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :-------------------------------------------------: | :------------------------------------------------: |
| Sub1    | [pose_resnet_50](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/res50_jhmdb_sub1_256x256.py) |  256x256   | 93.3 | 83.2 | 74.4 | 72.7 | 85.0 | 81.2 | 78.9 | 81.9 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub1_256x256-932cb3b4_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub1_256x256_20201122.log.json) |
| Sub2    | [pose_resnet_50](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/res50_jhmdb_sub2_256x256.py) |  256x256   | 94.1 | 74.9 | 64.5 | 62.5 | 77.9 | 71.9 | 78.6 | 75.5 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub2_256x256-83d606f7_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub2_256x256_20201122.log.json) |
| Sub3    | [pose_resnet_50](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/res50_jhmdb_sub3_256x256.py) |  256x256   | 97.0 | 82.2 | 74.9 | 70.7 | 84.7 | 83.7 | 84.2 | 82.9 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub3_256x256-c4ec1a0b_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub3_256x256_20201122.log.json) |
| Average |                   pose_resnet_50                   |  256x256   | 94.8 | 80.1 | 71.3 | 68.6 | 82.5 | 78.9 | 80.6 | 80.1 |                          -                          |                         -                          |
| Sub1    | [pose_resnet_50 (2 Deconv.)](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/res50_2deconv_jhmdb_sub1_256x256.py) |  256x256   | 92.4 | 80.6 | 73.2 | 70.5 | 82.3 | 75.4 | 75.0 | 79.2 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub1_256x256-f0574a52_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub1_256x256_20201122.log.json) |
| Sub2    | [pose_resnet_50 (2 Deconv.)](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/res50_2deconv_jhmdb_sub2_256x256.py) |  256x256   | 93.4 | 73.6 | 63.8 | 60.5 | 75.1 | 68.4 | 75.5 | 73.7 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub2_256x256-f63af0ff_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub2_256x256_20201122.log.json) |
| Sub3    | [pose_resnet_50 (2 Deconv.)](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/res50_2deconv_jhmdb_sub3_256x256.py) |  256x256   | 96.1 | 81.2 | 72.6 | 67.9 | 83.6 | 80.9 | 81.5 | 81.2 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub3_256x256-c4bc2ddb_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub3_256x256_20201122.log.json) |
| Average |             pose_resnet_50 (2 Deconv.)             |  256x256   | 94.0 | 78.5 | 69.9 | 66.3 | 80.3 | 74.9 | 77.3 | 78.0 |                          -                          |                         -                          |
