# Body 2D Keypoint

<hr/>
<br/><br/>

## JHMDB Dataset

<br/>

### Body 2d Keypoint + Resnet + JHMDB on JHMDB

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

<br/>

### Body 2d Keypoint + CPM + JHMDB on JHMDB

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2016/html/Wei_Convolutional_Pose_Machines_CVPR_2016_paper.html">CPM (CVPR'2016)</a></summary>

```bibtex
@inproceedings{wei2016convolutional,
  title={Convolutional pose machines},
  author={Wei, Shih-En and Ramakrishna, Varun and Kanade, Takeo and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={4724--4732},
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

The models are pre-trained on MPII dataset only. NO test-time augmentation (multi-scale /rotation testing) is used.

- Normalized by Person Size

| Split   |                        Arch                        | Input Size | Head | Sho  | Elb  | Wri  | Hip  | Knee | Ank  | Mean |                        ckpt                         |                        log                         |
| :------ | :------------------------------------------------: | :--------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :-------------------------------------------------: | :------------------------------------------------: |
| Sub1    | [cpm](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/cpm_jhmdb_sub1_368x368.py) |  368x368   | 96.1 | 91.9 | 81.0 | 78.9 | 96.6 | 90.8 | 87.3 | 89.5 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub1_368x368-2d2585c9_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub1_368x368_20201122.log.json) |
| Sub2    | [cpm](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/cpm_jhmdb_sub2_368x368.py) |  368x368   | 98.1 | 93.6 | 77.1 | 70.9 | 94.0 | 89.1 | 84.7 | 87.4 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub2_368x368-fc742f1f_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub2_368x368_20201122.log.json) |
| Sub3    | [cpm](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/cpm_jhmdb_sub3_368x368.py) |  368x368   | 97.9 | 94.9 | 87.3 | 84.0 | 98.6 | 94.4 | 86.2 | 92.4 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub3_368x368-49337155_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub3_368x368_20201122.log.json) |
| Average |                        cpm                         |  368x368   | 97.4 | 93.5 | 81.5 | 77.9 | 96.4 | 91.4 | 86.1 | 89.8 |                          -                          |                         -                          |

- Normalized by Torso Size

| Split   |                        Arch                        | Input Size | Head | Sho  | Elb  | Wri  | Hip  | Knee | Ank  | Mean |                        ckpt                         |                        log                         |
| :------ | :------------------------------------------------: | :--------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :-------------------------------------------------: | :------------------------------------------------: |
| Sub1    | [cpm](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/cpm_jhmdb_sub1_368x368.py) |  368x368   | 89.0 | 63.0 | 54.0 | 54.9 | 68.2 | 63.1 | 61.2 | 66.0 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub1_368x368-2d2585c9_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub1_368x368_20201122.log.json) |
| Sub2    | [cpm](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/cpm_jhmdb_sub2_368x368.py) |  368x368   | 90.3 | 57.9 | 46.8 | 44.3 | 60.8 | 58.2 | 62.4 | 61.1 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub2_368x368-fc742f1f_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub2_368x368_20201122.log.json) |
| Sub3    | [cpm](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/cpm_jhmdb_sub3_368x368.py) |  368x368   | 91.0 | 72.6 | 59.9 | 54.0 | 73.2 | 68.5 | 65.8 | 70.3 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub3_368x368-49337155_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub3_368x368_20201122.log.json) |
| Average |                        cpm                         |  368x368   | 90.1 | 64.5 | 53.6 | 51.1 | 67.4 | 63.3 | 63.1 | 65.7 |                          -                          |                         -                          |

<hr/>
<br/><br/>

## Aic Dataset

<br/>

### Body 2d Keypoint + Resnet + Aic on Aic

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
<summary align="right"><a href="https://arxiv.org/abs/1711.06475">AI Challenger (ArXiv'2017)</a></summary>

```bibtex
@article{wu2017ai,
  title={Ai challenger: A large-scale dataset for going deeper in image understanding},
  author={Wu, Jiahong and Zheng, He and Zhao, Bo and Li, Yixin and Yan, Baoming and Liang, Rui and Wang, Wenjia and Zhou, Shipei and Lin, Guosen and Fu, Yanwei and others},
  journal={arXiv preprint arXiv:1711.06475},
  year={2017}
}
```

</details>

Results on AIC val set with ground-truth bounding boxes

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_resnet_101](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/aic/res101_aic_256x192.py) |  256x192   | 0.294 |      0.736      |      0.172      | 0.337 |      0.762      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_aic_256x192-79b35445_20200826.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_aic_256x192_20200826.log.json) |

<br/>

### Body 2d Keypoint + Hrnet + Aic on Aic

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
<summary align="right"><a href="https://arxiv.org/abs/1711.06475">AI Challenger (ArXiv'2017)</a></summary>

```bibtex
@article{wu2017ai,
  title={Ai challenger: A large-scale dataset for going deeper in image understanding},
  author={Wu, Jiahong and Zheng, He and Zhao, Bo and Li, Yixin and Yan, Baoming and Liang, Rui and Wang, Wenjia and Zhou, Shipei and Lin, Guosen and Fu, Yanwei and others},
  journal={arXiv preprint arXiv:1711.06475},
  year={2017}
}
```

</details>

Results on AIC val set with ground-truth bounding boxes

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_hrnet_w32](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/aic/hrnet_w32_aic_256x192.py) |  256x192   | 0.323 |      0.761      |      0.218      | 0.366 |      0.789      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_aic_256x192-30a4e465_20200826.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_aic_256x192_20200826.log.json) |

<hr/>
<br/><br/>

## Crowdpose Dataset

<br/>

### Body 2d Keypoint + Resnet + Crowdpose on Crowdpose

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
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Li_CrowdPose_Efficient_Crowded_Scenes_Pose_Estimation_and_a_New_Benchmark_CVPR_2019_paper.html">CrowdPose (CVPR'2019)</a></summary>

```bibtex
@article{li2018crowdpose,
  title={CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark},
  author={Li, Jiefeng and Wang, Can and Zhu, Hao and Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu},
  journal={arXiv preprint arXiv:1812.00324},
  year={2018}
}
```

</details>

Results on CrowdPose test with [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) human detector

| Arch                                           | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup>    |                      ckpt                      | log |
| :--------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :----: | :----: | :----------------------------------------------: | :--------------------------------------------: | :-: |
| [pose_resnet_50](/configs/body_2d_keypoint/topdown_heatmap/crowdpose/td-hm_res50_8xb64-210e_crowdpose-256x192.py) |  256x192   | 0.637 |      0.808      |      0.692      | 0.785  | 0.738  | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_crowdpose_256x192-c6a526b6_20201227.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_crowdpose_256x192_20201227.log.json) |     |
| [pose_resnet_101](/configs/body_2d_keypoint/topdown_heatmap/crowdpose/td-hm_res101_8xb64-210e_crowdpose-256x192.py) |  256x192   | 0.647 |      0.81       |      0.703      | 0.796  | 0.746  | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_crowdpose_256x192-8f5870f4_20201227.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_crowdpose_256x192_20201227.log.json) |     |
| [pose_resnet_101](/configs/body_2d_keypoint/topdown_heatmap/crowdpose/td-hm_res101_8xb64-210e_crowdpose-320x256.py) |  320x256   | 0.661 |      0.821      |      0.714      |  0.8   | 0.759  | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_crowdpose_320x256-c88c512a_20201227.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_crowdpose_320x256_20201227.log.json) |     |
| [pose_resnet_152](/configs/body_2d_keypoint/topdown_heatmap/crowdpose/td-hm_res152_8xb64-210e_crowdpose-256x192.py) |  256x192   | 0.656 |      0.818      |      0.712      | 0.803  | 0.754  | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_crowdpose_256x192-dbd49aba_20201227.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_crowdpose_256x192_20201227.log.json) |     |

<br/>

### Body 2d Keypoint + Hrnet + Crowdpose on Crowdpose

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
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Li_CrowdPose_Efficient_Crowded_Scenes_Pose_Estimation_and_a_New_Benchmark_CVPR_2019_paper.html">CrowdPose (CVPR'2019)</a></summary>

```bibtex
@article{li2018crowdpose,
  title={CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark},
  author={Li, Jiefeng and Wang, Can and Zhu, Hao and Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu},
  journal={arXiv preprint arXiv:1812.00324},
  year={2018}
}
```

</details>

Results on CrowdPose test with [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) human detector

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_hrnet_w32](/configs/body_2d_keypoint/topdown_heatmap/crowdpose/td-hm_hrnet-w32_8xb64-210e_crowdpose-256x192.py) |  256x192   | 0.675 |      0.825      |      0.729      | 0.816 |      0.769      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_crowdpose_256x192-960be101_20201227.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_crowdpose_256x192_20201227.log.json) |

<hr/>
<br/><br/>

## Mpii Dataset

<br/>

### Body 2d Keypoint + Litehrnet + Mpii on Mpii

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2104.06403">LiteHRNet (CVPR'2021)</a></summary>

```bibtex
@inproceedings{Yulitehrnet21,
  title={Lite-HRNet: A Lightweight High-Resolution Network},
  author={Yu, Changqian and Xiao, Bin and Gao, Changxin and Yuan, Lu and Zhang, Lei and Sang, Nong and Wang, Jingdong},
  booktitle={CVPR},
  year={2021}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch                                                        | Input Size | Mean  | Mean@0.1 |                            ckpt                             |                             log                             |
| :---------------------------------------------------------- | :--------: | :---: | :------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| [LiteHRNet-18](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_litehrnet-18_8xb64-210e_mpii-256x256.py) |  256x256   | 0.859 |   0.26   | [ckpt](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet18_mpii_256x256-cabd7984_20210623.pth) | [log](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet18_mpii_256x256_20210623.log.json) |
| [LiteHRNet-30](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_litehrnet-30_8xb64-210e_mpii-256x256.py) |  256x256   | 0.869 |  0.271   | [ckpt](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet30_mpii_256x256-faae8bd8_20210622.pth) | [log](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet30_mpii_256x256_20210622.log.json) |

<br/>

### Body 2d Keypoint + Scnet + Mpii on Mpii

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
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch                                                        | Input Size | Mean  | Mean@0.1 |                            ckpt                             |                             log                             |
| :---------------------------------------------------------- | :--------: | :---: | :------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| [pose_scnet_50](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_scnet50_8xb64-210e_mpii-256x256.py) |  256x256   | 0.888 |   0.29   | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_mpii_256x256-a54b6af5_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_mpii_256x256_20200812.log.json) |
| [pose_scnet_101](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_scnet101_8xb64-210e_mpii-256x256.py) |  256x256   | 0.887 |  0.293   | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_mpii_256x256-b4c2d184_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_mpii_256x256_20200812.log.json) |

<br/>

### Body 2d Keypoint + Seresnet + Mpii on Mpii

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper">SEResNet (CVPR'2018)</a></summary>

```bibtex
@inproceedings{hu2018squeeze,
  title={Squeeze-and-excitation networks},
  author={Hu, Jie and Shen, Li and Sun, Gang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={7132--7141},
  year={2018}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch                                                        | Input Size | Mean  | Mean@0.1 |                            ckpt                             |                             log                             |
| :---------------------------------------------------------- | :--------: | :---: | :------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| [pose_seresnet_50](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_seresnet50_8xb64-210e_mpii-256x256.py) |  256x256   | 0.884 |  0.292   | [ckpt](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet50_mpii_256x256-1bb21f79_20200927.pth) | [log](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet50_mpii_256x256_20200927.log.json) |
| [pose_seresnet_101](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_seresnet101_8xb64-210e_mpii-256x256.py) |  256x256   | 0.884 |  0.295   | [ckpt](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet101_mpii_256x256-0ba14ff5_20200927.pth) | [log](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet101_mpii_256x256_20200927.log.json) |
| [pose_seresnet_152\*](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_seresnet152_8xb64-210e_mpii-256x256.py) |  256x256   | 0.884 |  0.287   | [ckpt](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet152_mpii_256x256-6ea1e774_20200927.pth) | [log](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet152_mpii_256x256_20200927.log.json) |

Note that * means without imagenet pre-training.

<br/>

### Body 2d Keypoint + Hrnet + Dark + Mpii on Mpii

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

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.html">DarkPose (CVPR'2020)</a></summary>

```bibtex
@inproceedings{zhang2020distribution,
  title={Distribution-aware coordinate representation for human pose estimation},
  author={Zhang, Feng and Zhu, Xiatian and Dai, Hanbin and Ye, Mao and Zhu, Ce},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7093--7102},
  year={2020}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch                                                        | Input Size | Mean  | Mean@0.1 |                            ckpt                             |                             log                             |
| :---------------------------------------------------------- | :--------: | :---: | :------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| [pose_hrnet_w32_dark](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_hrnet-w32_dark-8xb64-210e_mpii-256x256.py) |  256x256   | 0.904 |  0.354   | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256_dark-f1601c5b_20200927.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256_dark_20200927.log.json) |
| [pose_hrnet_w48_dark](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_hrnet-w48_dark-8xb64-210e_mpii-256x256.py) |  256x256   | 0.905 |   0.36   | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_mpii_256x256_dark-0decd39f_20200927.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_mpii_256x256_dark_20200927.log.json) |

<br/>

### Body 2d Keypoint + Resnetv1d + Mpii on Mpii

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.html">ResNetV1D (CVPR'2019)</a></summary>

```bibtex
@inproceedings{he2019bag,
  title={Bag of tricks for image classification with convolutional neural networks},
  author={He, Tong and Zhang, Zhi and Zhang, Hang and Zhang, Zhongyue and Xie, Junyuan and Li, Mu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={558--567},
  year={2019}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch                                                        | Input Size | Mean  | Mean@0.1 |                            ckpt                             |                             log                             |
| :---------------------------------------------------------- | :--------: | :---: | :------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| [pose_resnetv1d_50](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_resnetv1d50_8xb64-210e_mpii-256x256.py) |  256x256   | 0.881 |   0.29   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_mpii_256x256-2337a92e_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_mpii_256x256_20200812.log.json) |
| [pose_resnetv1d_101](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_resnetv1d101_8xb64-210e_mpii-256x256.py) |  256x256   | 0.883 |  0.295   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_mpii_256x256-2851d710_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_mpii_256x256_20200812.log.json) |
| [pose_resnetv1d_152](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_resnetv1d152_8xb64-210e_mpii-256x256.py) |  256x256   | 0.888 |   0.3    | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_mpii_256x256-8b10a87c_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_mpii_256x256_20200812.log.json) |

<br/>

### Body 2d Keypoint + Resnext + Mpii on Mpii

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html">ResNext (CVPR'2017)</a></summary>

```bibtex
@inproceedings{xie2017aggregated,
  title={Aggregated residual transformations for deep neural networks},
  author={Xie, Saining and Girshick, Ross and Doll{\'a}r, Piotr and Tu, Zhuowen and He, Kaiming},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1492--1500},
  year={2017}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch                                                        | Input Size | Mean  | Mean@0.1 |                            ckpt                             |                             log                             |
| :---------------------------------------------------------- | :--------: | :---: | :------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| [pose_resnext_152](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_resnext152_8xb64-210e_mpii-256x256.py) |  256x256   | 0.887 |  0.294   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_mpii_256x256-df302719_20200927.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_mpii_256x256_20200927.log.json) |

<br/>

### Body 2d Keypoint + Shufflenetv2 + Mpii on Mpii

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html">ShufflenetV2 (ECCV'2018)</a></summary>

```bibtex
@inproceedings{ma2018shufflenet,
  title={Shufflenet v2: Practical guidelines for efficient cnn architecture design},
  author={Ma, Ningning and Zhang, Xiangyu and Zheng, Hai-Tao and Sun, Jian},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={116--131},
  year={2018}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch                                                        | Input Size | Mean  | Mean@0.1 |                            ckpt                             |                             log                             |
| :---------------------------------------------------------- | :--------: | :---: | :------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| [pose_shufflenetv2](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_shufflenetv2_8xb64-210e_mpii-256x256.py) |  256x256   | 0.828 |  0.205   | [ckpt](https://download.openmmlab.com/mmpose/top_down/shufflenetv2/shufflenetv2_mpii_256x256-4fb9df2d_20200925.pth) | [log](https://download.openmmlab.com/mmpose/top_down/shufflenetv2/shufflenetv2_mpii_256x256_20200925.log.json) |

<br/>

### Body 2d Keypoint + CPM + Mpii on Mpii

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2016/html/Wei_Convolutional_Pose_Machines_CVPR_2016_paper.html">CPM (CVPR'2016)</a></summary>

```bibtex
@inproceedings{wei2016convolutional,
  title={Convolutional pose machines},
  author={Wei, Shih-En and Ramakrishna, Varun and Kanade, Takeo and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={4724--4732},
  year={2016}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch                                                        | Input Size | Mean  | Mean@0.1 |                            ckpt                             |                             log                             |
| :---------------------------------------------------------- | :--------: | :---: | :------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| [cpm](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_cpm_8xb64-210e_mpii-368x368.py) |  368x368   | 0.876 |  0.285   | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_mpii_368x368-116e62b8_20200822.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_mpii_368x368_20200822.log.json) |

<br/>

### Body 2d Keypoint + Mobilenetv2 + Mpii on Mpii

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html">MobilenetV2 (CVPR'2018)</a></summary>

```bibtex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch                                                        | Input Size | Mean  | Mean@0.1 |                            ckpt                             |                             log                             |
| :---------------------------------------------------------- | :--------: | :---: | :------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| \[pose_mobilenetv2\](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_mobilenetv2_8xb64-210e_mpii-256x256.py |  256x256   | 0.854 |  0.234   | [ckpt](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_mpii_256x256-e068afa7_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_mpii_256x256_20200812.log.json) |

<br/>

### Body 2d Keypoint + Shufflenetv1 + Mpii on Mpii

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.html">ShufflenetV1 (CVPR'2018)</a></summary>

```bibtex
@inproceedings{zhang2018shufflenet,
  title={Shufflenet: An extremely efficient convolutional neural network for mobile devices},
  author={Zhang, Xiangyu and Zhou, Xinyu and Lin, Mengxiao and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={6848--6856},
  year={2018}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch                                                        | Input Size | Mean  | Mean@0.1 |                            ckpt                             |                             log                             |
| :---------------------------------------------------------- | :--------: | :---: | :------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| [pose_shufflenetv1](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_shufflenetv1_8xb64-210e_mpii-256x256.py) |  256x256   | 0.824 |  0.195   | [ckpt](https://download.openmmlab.com/mmpose/top_down/shufflenetv1/shufflenetv1_mpii_256x256-dcc1c896_20200925.pth) | [log](https://download.openmmlab.com/mmpose/top_down/shufflenetv1/shufflenetv1_mpii_256x256_20200925.log.json) |

<br/>

### Body 2d Keypoint + Hourglass + Mpii on Mpii

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
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch                                                        | Input Size | Mean  | Mean@0.1 |                            ckpt                             |                             log                             |
| :---------------------------------------------------------- | :--------: | :---: | :------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| [pose_hourglass_52](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_hourglass52_8xb64-210e_mpii-256x256.py) |  256x256   | 0.889 |  0.317   | [ckpt](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_mpii_256x256-ae358435_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_mpii_256x256_20200812.log.json) |
| [pose_hourglass_52](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_hourglass52_8xb32-210e_mpii-384x384.py) |  384x384   | 0.894 |  0.367   | [ckpt](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_mpii_384x384-04090bc3_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_mpii_384x384_20200812.log.json) |

<br/>

### Body 2d Keypoint + Hrnet + Mpii on Mpii

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
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch                                                        | Input Size | Mean  | Mean@0.1 |                            ckpt                             |                             log                             |
| :---------------------------------------------------------- | :--------: | :---: | :------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| [pose_hrnet_w32](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_hrnet-w32_8xb64-210e_mpii-256x256.py) |  256x256   |  0.9  |  0.334   | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256-6c4f923f_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256_20200812.log.json) |
| [pose_hrnet_w48](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_hrnet-w48_8xb64-210e_mpii-256x256.py) |  256x256   | 0.901 |  0.337   | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_mpii_256x256-92cab7bd_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_mpii_256x256_20200812.log.json) |

<br/>

### Body 2d Keypoint + Resnet + Mpii on Mpii

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.html">DeepPose (CVPR'2014)</a></summary>

```bibtex
@inproceedings{toshev2014deeppose,
  title={Deeppose: Human pose estimation via deep neural networks},
  author={Toshev, Alexander and Szegedy, Christian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1653--1660},
  year={2014}
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
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch                                                        | Input Size | Mean  | Mean@0.1 |                            ckpt                             |                             log                             |
| :---------------------------------------------------------- | :--------: | :---: | :------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| [deeppose_resnet_50](/configs/body_2d_keypoint/topdown_regression/mpii/td-reg_res50_8xb64-210e_mpii-256x256.py) |  256x256   | 0.826 |   0.18   | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_mpii_256x256-c63cd0b6_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_mpii_256x256_20210203.log.json) |
| [deeppose_resnet_101](/configs/body_2d_keypoint/topdown_regression/mpii/td-reg_res101_8xb64-210e_mpii-256x256.py) |  256x256   | 0.841 |   0.2    | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_mpii_256x256-87516a90_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_mpii_256x256_20210205.log.json) |
| [deeppose_resnet_152](/configs/body_2d_keypoint/topdown_regression/mpii/td-reg_res152_8xb64-210e_mpii-256x256.py) |  256x256   | 0.85  |  0.208   | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_mpii_256x256-15f5e6f9_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_mpii_256x256_20210205.log.json) |

<br/>

### Body 2d Keypoint + Resnet + Rle + Mpii on Mpii

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.html">DeepPose (CVPR'2014)</a></summary>

```bibtex
@inproceedings{toshev2014deeppose,
  title={Deeppose: Human pose estimation via deep neural networks},
  author={Toshev, Alexander and Szegedy, Christian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1653--1660},
  year={2014}
}
```

</details>

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2107.11291">RLE (ICCV'2021)</a></summary>

```bibtex
@inproceedings{li2021human,
  title={Human pose regression with residual log-likelihood estimation},
  author={Li, Jiefeng and Bian, Siyuan and Zeng, Ailing and Wang, Can and Pang, Bo and Liu, Wentao and Lu, Cewu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11025--11034},
  year={2021}
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
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch                                                        | Input Size | Mean  | Mean@0.1 |                            ckpt                             |                             log                             |
| :---------------------------------------------------------- | :--------: | :---: | :------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| [deeppose_resnet_50_rle](/configs/body_2d_keypoint/topdown_regression/mpii/td-reg_res50_rle-8xb64-210e_mpii-256x256.py) |  256x256   | 0.861 |  0.277   | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_mpii_256x256_rle-5f92a619_20220504.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_mpii_256x256_rle_20220504.log.json) |

<hr/>
<br/><br/>

## Posetrack18 Dataset

<br/>

### Body 2d Keypoint + Resnet + Posetrack18 on Posetrack18

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

Results on PoseTrack2018 val with ground-truth bounding boxes

| Arch                                                 | Input Size | Head | Shou | Elb  | Wri  | Hip  | Knee | Ankl | Total |                         ckpt                          |                         log                          |
| :--------------------------------------------------- | :--------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :---: | :---------------------------------------------------: | :--------------------------------------------------: |
| [pose_resnet_50](/configs/body_2d_keypoint/topdown_heatmap/posetrack18/td-hm_res50_8xb64-20e_posetrack18-256x192.py) |  256x192   | 86.5 | 87.7 | 82.5 | 75.8 | 80.1 | 78.8 | 74.2 | 81.2  | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_posetrack18_256x192-a62807c7_20201028.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_posetrack18_256x192_20201028.log.json) |

The models are first pre-trained on COCO dataset, and then fine-tuned on PoseTrack18.

Results on PoseTrack2018 val with [MMDetection](https://github.com/open-mmlab/mmdetection) pre-trained [Cascade R-CNN](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth) (X-101-64x4d-FPN) human detector

| Arch                                                 | Input Size | Head | Shou | Elb  | Wri  | Hip  | Knee | Ankl | Total |                         ckpt                          |                         log                          |
| :--------------------------------------------------- | :--------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :---: | :---------------------------------------------------: | :--------------------------------------------------: |
| [pose_resnet_50](/configs/body_2d_keypoint/topdown_heatmap/posetrack18/td-hm_res50_8xb64-20e_posetrack18-256x192.py) |  256x192   | 86.5 | 87.7 | 82.5 | 75.8 | 80.1 | 78.8 | 74.2 | 81.2  | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_posetrack18_256x192-a62807c7_20201028.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_posetrack18_256x192_20201028.log.json) |

The models are first pre-trained on COCO dataset, and then fine-tuned on PoseTrack18.

<br/>

### Body 2d Keypoint + Hrnet + Posetrack18 on Posetrack18

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

Results on PoseTrack2018 val with ground-truth bounding boxes

| Arch                                                 | Input Size | Head | Shou | Elb  | Wri  | Hip  | Knee | Ankl | Total |                         ckpt                          |                         log                          |
| :--------------------------------------------------- | :--------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :---: | :---------------------------------------------------: | :--------------------------------------------------: |
| [pose_hrnet_w32](/configs/body_2d_keypoint/topdown_heatmap/posetrack18/td-hm_hrnet-w32_8xb64-20e_posetrack18-256x192.py) |  256x192   | 86.2 |  89  | 84.5 | 79.2 | 82.3 | 82.5 | 78.7 | 83.4  | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_256x192-1ee951c4_20201028.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_256x192_20201028.log.json) |
| [pose_hrnet_w32](/configs/body_2d_keypoint/topdown_heatmap/posetrack18/td-hm_hrnet-w32_8xb64-20e_posetrack18-384x288.py) |  384x288   | 87.1 |  89  | 85.1 | 80.2 | 80.6 | 82.8 | 79.6 | 83.7  | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_384x288-806f00a3_20211130.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_384x288_20211130.log.json) |
| [pose_hrnet_w48](/configs/body_2d_keypoint/topdown_heatmap/posetrack18/td-hm_hrnet-w48_8xb64-20e_posetrack18-256x192.py) |  256x192   | 88.3 | 90.2 |  86  |  81  | 80.7 | 83.3 | 80.6 | 84.6  | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_256x192-b5d9b3f1_20211130.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_256x192_20211130.log.json) |
| [pose_hrnet_w48](/configs/body_2d_keypoint/topdown_heatmap/posetrack18/td-hm_hrnet-w48_8xb64-20e_posetrack18-384x288.py) |  384x288   | 87.8 |  9   | 86.2 | 81.3 |  81  | 83.4 | 80.9 | 84.6  | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_384x288-5fd6d3ff_20211130.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_384x288_20211130.log.json) |

The models are first pre-trained on COCO dataset, and then fine-tuned on PoseTrack18.

Results on PoseTrack2018 val with [MMDetection](https://github.com/open-mmlab/mmdetection) pre-trained [Cascade R-CNN](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth) (X-101-64x4d-FPN) human detector

| Arch                                                 | Input Size | Head | Shou | Elb  | Wri  | Hip  | Knee | Ankl | Total |                         ckpt                          |                         log                          |
| :--------------------------------------------------- | :--------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :---: | :---------------------------------------------------: | :--------------------------------------------------: |
| [pose_hrnet_w32](/configs/body_2d_keypoint/topdown_heatmap/posetrack18/td-hm_hrnet-w32_8xb64-20e_posetrack18-256x192.py) |  256x192   | 86.2 |  89  | 84.5 | 79.2 | 82.3 | 82.5 | 78.7 | 83.4  | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_256x192-1ee951c4_20201028.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_256x192_20201028.log.json) |
| [pose_hrnet_w32](/configs/body_2d_keypoint/topdown_heatmap/posetrack18/td-hm_hrnet-w32_8xb64-20e_posetrack18-384x288.py) |  384x288   | 87.1 |  89  | 85.1 | 80.2 | 80.6 | 82.8 | 79.6 | 83.7  | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_384x288-806f00a3_20211130.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_384x288_20211130.log.json) |
| [pose_hrnet_w48](/configs/body_2d_keypoint/topdown_heatmap/posetrack18/td-hm_hrnet-w48_8xb64-20e_posetrack18-256x192.py) |  256x192   | 88.3 | 90.2 |  86  |  81  | 80.7 | 83.3 | 80.6 | 84.6  | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_256x192-b5d9b3f1_20211130.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_256x192_20211130.log.json) |
| [pose_hrnet_w48](/configs/body_2d_keypoint/topdown_heatmap/posetrack18/td-hm_hrnet-w48_8xb64-20e_posetrack18-384x288.py) |  384x288   | 87.8 |  9   | 86.2 | 81.3 |  81  | 83.4 | 80.9 | 84.6  | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_384x288-5fd6d3ff_20211130.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_384x288_20211130.log.json) |

The models are first pre-trained on COCO dataset, and then fine-tuned on PoseTrack18.

<hr/>
<br/><br/>

## Coco Dataset

<br/>

### Body 2d Keypoint + Resnet + Coco on Coco

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.html">DeepPose (CVPR'2014)</a></summary>

```bibtex
@inproceedings{toshev2014deeppose,
  title={Deeppose: Human pose estimation via deep neural networks},
  author={Toshev, Alexander and Szegedy, Christian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1653--1660},
  year={2014}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [deeppose_resnet_50](/configs/body_2d_keypoint/topdown_regression/coco/td-reg_res50_8xb64-210e_coco-256x192.py) |  256x192   | 0.528 |      0.817      |      0.589      | 0.639 |      0.888      | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_coco_256x192-f6de6c0e_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_coco_256x192_20210205.log.json) |
| [deeppose_resnet_101](/configs/body_2d_keypoint/topdown_regression/coco/td-reg_res101_8xb64-210e_coco-256x192.py) |  256x192   | 0.562 |      0.831      |      0.629      | 0.67  |       0.9       | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_coco_256x192-2f247111_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_coco_256x192_20210205.log.json) |
| [deeppose_resnet_152](/configs/body_2d_keypoint/topdown_regression/coco/td-reg_res152_8xb64-210e_coco-256x192.py) |  256x192   | 0.584 |      0.842      |      0.659      | 0.688 |      0.907      | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_256x192-7df89a88_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_256x192_20210205.log.json) |

<br/>

### Body 2d Keypoint + MSPN + Coco on Coco

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1901.00148">MSPN (ArXiv'2019)</a></summary>

```bibtex
@article{li2019rethinking,
  title={Rethinking on Multi-Stage Networks for Human Pose Estimation},
  author={Li, Wenbo and Wang, Zhicheng and Yin, Binyi and Peng, Qixiang and Du, Yuming and Xiao, Tianzi and Yu, Gang and Lu, Hongtao and Wei, Yichen and Sun, Jian},
  journal={arXiv preprint arXiv:1901.00148},
  year={2019}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [mspn_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_mspn50_8xb32-210e_coco-256x192.py) |  256x192   | 0.723 |      0.895      |      0.794      | 0.788 |      0.934      | [ckpt](https://download.openmmlab.com/mmpose/top_down/mspn/mspn50_coco_256x192-8fbfb5d0_20201123.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mspn/mspn50_coco_256x192_20201123.log.json) |
| [2xmspn_50](/configs/body_2d_keypoint/topdown_heatmap/coco/2xtd-hm_mspn50_8xb32-210e_coco-256x192.py) |  256x192   | 0.754 |      0.903      |      0.826      | 0.816 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/top_down/mspn/2xmspn50_coco_256x192-c8765a5c_20201123.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mspn/2xmspn50_coco_256x192_20201123.log.json) |
| [3xmspn_50](/configs/body_2d_keypoint/topdown_heatmap/coco/3xtd-hm_mspn50_8xb32-210e_coco-256x192.py) |  256x192   | 0.758 |      0.904      |      0.83       | 0.821 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/mspn/3xmspn50_coco_256x192-e348f18e_20201123.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mspn/3xmspn50_coco_256x192_20201123.log.json) |
| [4xmspn_50](/configs/body_2d_keypoint/topdown_heatmap/coco/4xtd-hm_mspn50_8xb32-210e_coco-256x192.py) |  256x192   | 0.765 |      0.906      |      0.835      | 0.826 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/mspn/4xmspn50_coco_256x192-7b837afb_20201123.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mspn/4xmspn50_coco_256x192_20201123.log.json) |

<br/>

### Body 2d Keypoint + Hrnet + Udp + Coco on Coco

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

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Huang_The_Devil_Is_in_the_Details_Delving_Into_Unbiased_Data_CVPR_2020_paper.html">UDP (CVPR'2020)</a></summary>

```bibtex
@InProceedings{Huang_2020_CVPR,
  author = {Huang, Junjie and Zhu, Zheng and Guo, Feng and Huang, Guan},
  title = {The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_hrnet_w32_udp](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-8xb64-210e_coco-256x192.py) |  256x192   | 0.76  |      0.906      |      0.827      | 0.811 |      0.945      | [ckpt](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_256x192_udp-aba0be42_20210220.pth) | [log](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_256x192_udp_20210220.log.json) |
| [pose_hrnet_w32_udp](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-8xb64-210e_coco-384x288.py) |  384x288   | 0.768 |      0.907      |      0.832      | 0.816 |      0.944      | [ckpt](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_384x288_udp-e97c1a0f_20210223.pth) | [log](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_384x288_udp_20210223.log.json) |
| [pose_hrnet_w48_udp](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192.py) |  256x192   | 0.767 |      0.907      |      0.834      | 0.818 |      0.946      | [ckpt](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w48_coco_256x192_udp-2554c524_20210223.pth) | [log](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w48_coco_256x192_udp_20210223.log.json) |
| [pose_hrnet_w48_udp](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py) |  384x288   | 0.773 |      0.91       |      0.835      | 0.82  |      0.945      | [ckpt](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w48_coco_384x288_udp-0f89c63e_20210223.pth) | [log](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w48_coco_384x288_udp_20210223.log.json) |
| [pose_hrnet_w32_udp_regress](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-regress-8xb64-210e_coco-256x192.py) |  256x192   | 0.758 |      0.908      |      0.823      | 0.812 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_256x192_udp_regress-be2dbba4_20210222.pth) | [log](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_256x192_udp_regress_20210222.log.json) |

Note that, UDP also adopts the unbiased encoding/decoding algorithm of [DARK](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#div-align-center-darkpose-cvpr-2020-div).

<br/>

### Body 2d Keypoint + Swin + Coco on Coco

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
<summary align="right"><a href="https://arxiv.org/abs/2103.14030">Swin (ICCV'2021)</a></summary>

```bibtex
@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10012--10022},
  year={2021}
}
```

</details>

<!-- [OTHERS] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html">FPN (CVPR'2017)</a></summary>

```bibtex
@inproceedings{lin2017feature,
  title={Feature pyramid networks for object detection},
  author={Lin, Tsung-Yi and Doll{\'a}r, Piotr and Girshick, Ross and He, Kaiming and Hariharan, Bharath and Belongie, Serge},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2117--2125},
  year={2017}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_swin_t](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_swin-t-p4-w7_8xb32-210e_coco-256x192.py) |  256x192   | 0.724 |      0.901      |      0.806      | 0.782 |      0.94       | [ckpt](https://download.openmmlab.com/mmpose/top_down/swin/swin_t_p4_w7_coco_256x192-eaefe010_20220503.pth) | [log](https://download.openmmlab.com/mmpose/top_down/swin/swin_t_p4_w7_coco_256x192_20220503.log.json) |
| [pose_swin_b](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_swin-b-p4-w7_8xb32-210e_coco-256x192.py) |  256x192   | 0.737 |      0.904      |      0.82       | 0.794 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/top_down/swin/swin_b_p4_w7_coco_256x192-7432be9e_20220705.pth) | [log](https://download.openmmlab.com/mmpose/top_down/swin/swin_b_p4_w7_coco_256x192_20220705.log.json) |
| [pose_swin_b](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_swin-b-p4-w7_8xb32-210e_coco-384x288.py) |  384x288   | 0.759 |      0.91       |      0.832      | 0.811 |      0.946      | [ckpt](https://download.openmmlab.com/mmpose/top_down/swin/swin_b_p4_w7_coco_384x288-3abf54f9_20220705.pth) | [log](https://download.openmmlab.com/mmpose/top_down/swin/swin_b_p4_w7_coco_384x288_20220705.log.json) |
| [pose_swin_l](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_swin-l-p4-w7_8xb32-210e_coco-256x192.py) |  256x192   | 0.743 |      0.906      |      0.821      | 0.798 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_256x192-642a89db_20220705.pth) | [log](https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_256x192_20220705.log.json) |
| [pose_swin_l](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_swin-l-p4-w7_8xb32-210e_coco-384x288.py) |  384x288   | 0.763 |      0.912      |      0.83       | 0.814 |      0.949      | [ckpt](https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_384x288-c36b7845_20220705.pth) | [log](https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_384x288_20220705.log.json) |

<br/>

### Body 2d Keypoint + Hrformer + Coco on Coco

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://proceedings.neurips.cc/paper/2021/hash/3bbfdde8842a5c44a0323518eec97cbe-Abstract.html">HRFormer (NIPS'2021)</a></summary>

```bibtex
@article{yuan2021hrformer,
  title={HRFormer: High-Resolution Vision Transformer for Dense Predict},
  author={Yuan, Yuhui and Fu, Rao and Huang, Lang and Lin, Weihong and Zhang, Chao and Chen, Xilin and Wang, Jingdong},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_hrformer_small](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrformer-small_8xb32-210e_coco-256x192.py) |  256x192   | 0.738 |      0.904      |      0.812      | 0.793 |      0.941      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_small_coco_256x192-5310d898_20220316.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_small_coco_256x192_20220316.log.json) |
| [pose_hrformer_small](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrformer-small_8xb32-210e_coco-384x288.py) |  384x288   | 0.757 |      0.905      |      0.824      | 0.807 |      0.941      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_small_coco_384x288-98d237ed_20220316.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_small_coco_384x288_20220316.log.json) |
| [pose_hrformer_base](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrformer-base_8xb32-210e_coco-256x192.py) |  256x192   | 0.754 |      0.906      |      0.827      | 0.807 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_base_coco_256x192-6f5f1169_20220316.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_base_coco_256x192_20220316.log.json) |
| [pose_hrformer_base](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrformer-base_8xb32-210e_coco-384x288.py) |  384x288   | 0.774 |      0.909      |      0.842      | 0.823 |      0.945      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_base_coco_384x288-ecf0758d_20220316.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_base_coco_256x192_20220316.log.json) |

<br/>

### Body 2d Keypoint + Hrnet + Dark + Coco on Coco

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

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.html">DarkPose (CVPR'2020)</a></summary>

```bibtex
@inproceedings{zhang2020distribution,
  title={Distribution-aware coordinate representation for human pose estimation},
  author={Zhang, Feng and Zhu, Xiatian and Dai, Hanbin and Ye, Mao and Zhu, Ce},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7093--7102},
  year={2020}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_hrnet_w32_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_dark-8xb64-210e_coco-256x192.py) |  256x192   | 0.757 |      0.908      |      0.822      | 0.808 |      0.944      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192_dark-07f147eb_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192_dark_20200812.log.json) |
| [pose_hrnet_w32_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_dark-8xb64-210e_coco-384x288.py) |  384x288   | 0.766 |      0.906      |      0.829      | 0.815 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_384x288_dark-307dafc2_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_384x288_dark_20210203.log.json) |
| [pose_hrnet_w48_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-256x192.py) |  256x192   | 0.764 |      0.907      |      0.829      | 0.814 |      0.944      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192_dark-8cba3197_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192_dark_20200812.log.json) |
| [pose_hrnet_w48_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288.py) |  384x288   | 0.772 |      0.91       |      0.837      | 0.821 |      0.946      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark_20210203.log.json) |

<br/>

### Body 2d Keypoint + Resnet + Dark + Coco on Coco

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

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.html">DarkPose (CVPR'2020)</a></summary>

```bibtex
@inproceedings{zhang2020distribution,
  title={Distribution-aware coordinate representation for human pose estimation},
  author={Zhang, Feng and Zhu, Xiatian and Dai, Hanbin and Ye, Mao and Zhu, Ce},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7093--7102},
  year={2020}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_resnet_50_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_dark-8xb64-210e_coco-256x192.py) |  256x192   | 0.724 |      0.898      |       0.8       | 0.777 |      0.936      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192_dark-43379d20_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192_dark_20200709.log.json) |
| [pose_resnet_50_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_dark-8xb64-210e_coco-384x288.py) |  384x288   | 0.734 |       0.9       |      0.801      | 0.785 |      0.937      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_384x288_dark-33d3e5e5_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_384x288_dark_20210203.log.json) |
| [pose_resnet_101_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res101_dark-8xb64-210e_coco-256x192.py) |  256x192   | 0.732 |      0.899      |      0.807      | 0.786 |      0.938      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192_dark-64d433e6_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192_dark_20200812.log.json) |
| [pose_resnet_101_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res101_dark-8xb64-210e_coco-384x288.py) |  384x288   | 0.749 |      0.902      |      0.817      | 0.799 |      0.938      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_384x288_dark-cb45c88d_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_384x288_dark_20210203.log.json) |
| [pose_resnet_152_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res152_dark-8xb32-210e_coco-256x192.py) |  256x192   | 0.744 |      0.904      |      0.821      | 0.797 |      0.941      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192_dark-ab4840d5_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192_dark_20200812.log.json) |
| [pose_resnet_152_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res152_dark-8xb32-210e_coco-384x288.py) |  384x288   | 0.756 |      0.909      |      0.826      | 0.805 |      0.944      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288_dark-d3b8ebd7_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288_dark_20210203.log.json) |

<br/>

### Body 2d Keypoint + VGG + Coco on Coco

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1409.1556">VGG (ICLR'2015)</a></summary>

```bibtex
@article{simonyan2014very,
  title={Very deep convolutional networks for large-scale image recognition},
  author={Simonyan, Karen and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1409.1556},
  year={2014}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [vgg](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_vgg16-bn_8xb64-210e_coco-256x192.py) |  256x192   | 0.699 |      0.89       |      0.769      | 0.754 |      0.927      | [ckpt](https://download.openmmlab.com/mmpose/top_down/vgg/vgg16_bn_coco_256x192-7e7c58d6_20210517.pth) | [log](https://download.openmmlab.com/mmpose/top_down/vgg/vgg16_bn_coco_256x192_20210517.log.json) |

<br/>

### Body 2d Keypoint + Alexnet + Coco on Coco

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf">AlexNet (NeurIPS'2012)</a></summary>

```bibtex
@inproceedings{krizhevsky2012imagenet,
  title={Imagenet classification with deep convolutional neural networks},
  author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  booktitle={Advances in neural information processing systems},
  pages={1097--1105},
  year={2012}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_alexnet](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_alexnet_8xb64-210e_coco-256x192.py) |  256x192   | 0.448 |      0.767      |      0.461      | 0.521 |      0.829      | [ckpt](https://download.openmmlab.com/mmpose/top_down/alexnet/alexnet_coco_256x192-a7b1fd15_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/alexnet/alexnet_coco_256x192_20200727.log.json) |

<br/>

### Body 2d Keypoint + Vipnas + Coco on Coco

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2105.10154">ViPNAS (CVPR'2021)</a></summary>

```bibtex
@article{xu2021vipnas,
  title={ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search},
  author={Xu, Lumin and Guan, Yingda and Jin, Sheng and Liu, Wentao and Qian, Chen and Luo, Ping and Ouyang, Wanli and Wang, Xiaogang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2021}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [S-ViPNAS-MobileNetV3](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_vipnas-mbv3_8xb64-210e_coco-256x192.py) |  256x192   |  0.7  |      0.887      |      0.777      | 0.757 |      0.929      | [ckpt](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_mbv3_coco_256x192-7018731a_20211122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_mbv3_coco_256x192_20211122.log.json) |
| [S-ViPNAS-Res50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_vipnas-res50_8xb64-210e_coco-256x192.py) |  256x192   | 0.711 |      0.893      |      0.79       | 0.769 |      0.934      | [ckpt](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_coco_256x192-cc43b466_20210624.pth) | [log](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_coco_256x192_20210624.log.json) |

<br/>

### Body 2d Keypoint + Resnext + Coco on Coco

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html">ResNext (CVPR'2017)</a></summary>

```bibtex
@inproceedings{xie2017aggregated,
  title={Aggregated residual transformations for deep neural networks},
  author={Xie, Saining and Girshick, Ross and Doll{\'a}r, Piotr and Tu, Zhuowen and He, Kaiming},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1492--1500},
  year={2017}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_resnext_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnext50_8xb64-210e_coco-256x192.py) |  256x192   | 0.715 |      0.897      |      0.791      | 0.771 |      0.935      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext50_coco_256x192-dcff15f6_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext50_coco_256x192_20200727.log.json) |
| [pose_resnext_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnext50_8xb64-210e_coco-384x288.py) |  384x288   | 0.724 |      0.899      |      0.794      | 0.777 |      0.936      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext50_coco_384x288-412c848f_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext50_coco_384x288_20200727.log.json) |
| [pose_resnext_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnext101_8xb64-210e_coco-256x192.py) |  256x192   | 0.726 |       0.9       |      0.801      | 0.781 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext101_coco_256x192-c7eba365_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext101_coco_256x192_20200727.log.json) |
| [pose_resnext_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnext101_8xb64-210e_coco-384x288.py) |  384x288   | 0.744 |      0.903      |      0.815      | 0.794 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext101_coco_384x288-f5eabcd6_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext101_coco_384x288_20200727.log.json) |
| [pose_resnext_152](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnext152_8xb32-210e_coco-256x192.py) |  256x192   | 0.73  |      0.903      |      0.808      | 0.785 |      0.94       | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_coco_256x192-102449aa_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_coco_256x192_20200727.log.json) |
| [pose_resnext_152](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnext152_8xb32-210e_coco-384x288.py) |  384x288   | 0.742 |      0.904      |      0.81       | 0.794 |      0.94       | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_coco_384x288-806176df_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_coco_384x288_20200727.log.json) |

<br/>

### Body 2d Keypoint + PVT + Coco on Coco

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2102.12122">PVT (ICCV'2021)</a></summary>

```bibtex
@inproceedings{wang2021pyramid,
  title={Pyramid vision transformer: A versatile backbone for dense prediction without convolutions},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Fan, Deng-Ping and Song, Kaitao and Liang, Ding and Lu, Tong and Luo, Ping and Shao, Ling},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={568--578},
  year={2021}
}
```

</details>

<details>
<summary align="right"><a href="https://arxiv.org/abs/2106.13797">PVTV2 (CVMJ'2022)</a></summary>

```bibtex
@article{wang2022pvt,
  title={PVT v2: Improved baselines with Pyramid Vision Transformer},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Fan, Deng-Ping and Song, Kaitao and Liang, Ding and Lu, Tong and Luo, Ping and Shao, Ling},
  journal={Computational Visual Media},
  pages={1--10},
  year={2022},
  publisher={Springer}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_pvt-s](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_pvt-s_8xb64-210e_coco-256x192.py) |  256x192   | 0.714 |      0.896      |      0.794      | 0.773 |      0.936      | [ckpt](https://download.openmmlab.com/mmpose/top_down/pvt/pvt_small_coco_256x192-4324a49d_20220501.pth) | [log](https://download.openmmlab.com/mmpose/top_down/pvt/pvt_small_coco_256x192_20220501.log.json) |
| [pose_pvtv2-b2](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_pvtv2-b2_8xb64-210e_coco-256x192.py) |  256x192   | 0.737 |      0.905      |      0.812      | 0.791 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/top_down/pvt/pvtv2_b2_coco_256x192-b4212737_20220501.pth) | [log](https://download.openmmlab.com/mmpose/top_down/pvt/pvtv2_b2_coco_256x192_20220501.log.json) |

<br/>

### Body 2d Keypoint + Hrnet + Augmentation + Coco on Coco

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

<!-- [OTHERS] -->

<details>
<summary align="right"><a href="https://www.mdpi.com/649002">Albumentations (Information'2020)</a></summary>

```bibtex
@article{buslaev2020albumentations,
  title={Albumentations: fast and flexible image augmentations},
  author={Buslaev, Alexander and Iglovikov, Vladimir I and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A},
  journal={Information},
  volume={11},
  number={2},
  pages={125},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [coarsedropout](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_coarsedropout-8xb64-210e_coco-256x192.py) |  256x192   | 0.753 |      0.908      |      0.822      | 0.805 |      0.944      | [ckpt](https://download.openmmlab.com/mmpose/top_down/augmentation/hrnet_w32_coco_256x192_coarsedropout-0f16a0ce_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/augmentation/hrnet_w32_coco_256x192_coarsedropout_20210320.log.json) |
| [gridmask](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_gridmask-8xb64-210e_coco-256x192.py) |  256x192   | 0.752 |      0.906      |      0.825      | 0.804 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/augmentation/hrnet_w32_coco_256x192_gridmask-868180df_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/augmentation/hrnet_w32_coco_256x192_gridmask_20210320.log.json) |
| [photometric](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_photometric-8xb64-210e_coco-256x192.py) |  256x192   | 0.754 |      0.908      |      0.825      | 0.805 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/augmentation/hrnet_w32_coco_256x192_photometric-308cf591_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/augmentation/hrnet_w32_coco_256x192_photometric_20210320.log.json) |

<br/>

### Body 2d Keypoint + CPM + Coco on Coco

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2016/html/Wei_Convolutional_Pose_Machines_CVPR_2016_paper.html">CPM (CVPR'2016)</a></summary>

```bibtex
@inproceedings{wei2016convolutional,
  title={Convolutional pose machines},
  author={Wei, Shih-En and Ramakrishna, Varun and Kanade, Takeo and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={4724--4732},
  year={2016}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [cpm](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_cpm_8xb64-210e_coco-256x192.py) |  256x192   | 0.623 |      0.858      |      0.706      | 0.685 |      0.902      | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_coco_256x192-aa4ba095_20200817.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_coco_256x192_20200817.log.json) |
| [cpm](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_cpm_8xb32-210e_coco-384x288.py) |  384x288   | 0.65  |      0.863      |      0.725      | 0.707 |      0.905      | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_coco_384x288-80feb4bc_20200821.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_coco_384x288_20200821.log.json) |

<br/>

### Body 2d Keypoint + Resnest + Coco on Coco

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2004.08955">ResNeSt (ArXiv'2020)</a></summary>

```bibtex
@article{zhang2020resnest,
  title={ResNeSt: Split-Attention Networks},
  author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
  journal={arXiv preprint arXiv:2004.08955},
  year={2020}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_resnest_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnest50_8xb64-210e_coco-256x192.py) |  256x192   | 0.72  |      0.899      |       0.8       | 0.775 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest50_coco_256x192-6e65eece_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest50_coco_256x192_20210320.log.json) |
| [pose_resnest_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnest50_8xb64-210e_coco-384x288.py) |  384x288   | 0.737 |       0.9       |      0.811      | 0.789 |      0.937      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest50_coco_384x288-dcd20436_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest50_coco_384x288_20210320.log.json) |
| [pose_resnest_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnest101_8xb64-210e_coco-256x192.py) |  256x192   | 0.725 |       0.9       |      0.807      | 0.781 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest101_coco_256x192-2ffcdc9d_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest101_coco_256x192_20210320.log.json) |
| [pose_resnest_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnest101_8xb64-210e_coco-384x288.py) |  384x288   | 0.745 |      0.905      |      0.818      | 0.798 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest101_coco_384x288-80660658_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest101_coco_384x288_20210320.log.json) |
| [pose_resnest_200](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnest200_8xb64-210e_coco-256x192.py) |  256x192   | 0.731 |      0.905      |      0.812      | 0.787 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest200_coco_256x192-db007a48_20210517.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest200_coco_256x192_20210517.log.json) |
| [pose_resnest_200](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnest200_8xb64-210e_coco-384x288.py) |  384x288   | 0.753 |      0.907      |      0.827      | 0.805 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest200_coco_384x288-b5bb76cb_20210517.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest200_coco_384x288_20210517.log.json) |
| [pose_resnest_269](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnest269_8xb32-210e_coco-256x192.py) |  256x192   | 0.737 |      0.907      |      0.819      | 0.792 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest269_coco_256x192-2a7882ac_20210517.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest269_coco_256x192_20210517.log.json) |
| [pose_resnest_269](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnest269_8xb32-210e_coco-384x288.py) |  384x288   | 0.754 |      0.908      |      0.828      | 0.805 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest269_coco_384x288-b142b9fb_20210517.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest269_coco_384x288_20210517.log.json) |

<br/>

### Body 2d Keypoint + Hrnet + Coco on Coco

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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_hrnet_w32](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py) |  256x192   | 0.746 |      0.904      |      0.819      | 0.799 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192_20200708.log.json) |
| [pose_hrnet_w32](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-384x288.py) |  384x288   | 0.76  |      0.906      |      0.83       | 0.81  |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_384x288-d9f0d786_20200708.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_384x288_20200708.log.json) |
| [pose_hrnet_w48](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py) |  256x192   | 0.756 |      0.907      |      0.825      | 0.806 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192_20200708.log.json) |
| [pose_hrnet_w48](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-384x288.py) |  384x288   | 0.767 |      0.91       |      0.831      | 0.816 |      0.946      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288-314c8528_20200708.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_20200708.log.json) |

<br/>

### Body 2d Keypoint + Scnet + Coco on Coco

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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_scnet_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_scnet50_8xb64-210e_coco-256x192.py) |  256x192   | 0.728 |      0.899      |      0.807      | 0.784 |      0.938      | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_coco_256x192-6920f829_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_coco_256x192_20200709.log.json) |
| [pose_scnet_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_scnet50_8xb64-210e_coco-384x288.py) |  384x288   | 0.751 |      0.906      |      0.818      | 0.802 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_coco_384x288-9cacd0ea_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_coco_384x288_20200709.log.json) |
| [pose_scnet_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_scnet101_8xb32-210e_coco-256x192.py) |  256x192   | 0.733 |      0.902      |      0.811      | 0.789 |      0.94       | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_coco_256x192-6d348ef9_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_coco_256x192_20200709.log.json) |
| [pose_scnet_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_scnet101_8xb48-210e_coco-384x288.py) |  384x288   | 0.752 |      0.906      |      0.823      | 0.804 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_coco_384x288-0b6e631b_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_coco_384x288_20200709.log.json) |

<br/>

### Body 2d Keypoint + Shufflenetv2 + Coco on Coco

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html">ShufflenetV2 (ECCV'2018)</a></summary>

```bibtex
@inproceedings{ma2018shufflenet,
  title={Shufflenet v2: Practical guidelines for efficient cnn architecture design},
  author={Ma, Ningning and Zhang, Xiangyu and Zheng, Hai-Tao and Sun, Jian},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={116--131},
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_shufflenetv2](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_shufflenetv2_8xb64-210e_coco-256x192.py) |  256x192   | 0.598 |      0.852      |      0.667      | 0.664 |      0.899      | [ckpt](https://download.openmmlab.com/mmpose/top_down/shufflenetv2/shufflenetv2_coco_256x192-0aba71c7_20200921.pth) | [log](https://download.openmmlab.com/mmpose/top_down/shufflenetv2/shufflenetv2_coco_256x192_20200921.log.json) |
| [pose_shufflenetv2](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_shufflenetv2_8xb64-210e_coco-384x288.py) |  384x288   | 0.636 |      0.865      |      0.703      | 0.697 |      0.91       | [ckpt](https://download.openmmlab.com/mmpose/top_down/shufflenetv2/shufflenetv2_coco_384x288-fb38ac3a_20200921.pth) | [log](https://download.openmmlab.com/mmpose/top_down/shufflenetv2/shufflenetv2_coco_384x288_20200921.log.json) |

<br/>

### Body 2d Keypoint + Hourglass + Coco on Coco

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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_hourglass_52](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hourglass52_8xb32-210e_coco-256x256.py) |  256x256   | 0.726 |      0.896      |      0.799      | 0.78  |      0.934      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_coco_256x256-4ec713ba_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_coco_256x256_20200709.log.json) |
| [pose_hourglass_52](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hourglass52_8xb32-210e_coco-384x384.py) |  384x384   | 0.746 |       0.9       |      0.812      | 0.797 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_coco_384x384-be91ba2b_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_coco_384x384_20200812.log.json) |

<br/>

### Body 2d Keypoint + Shufflenetv1 + Coco on Coco

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.html">ShufflenetV1 (CVPR'2018)</a></summary>

```bibtex
@inproceedings{zhang2018shufflenet,
  title={Shufflenet: An extremely efficient convolutional neural network for mobile devices},
  author={Zhang, Xiangyu and Zhou, Xinyu and Lin, Mengxiao and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={6848--6856},
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_shufflenetv1](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_shufflenetv1_8xb64-210e_coco-256x192.py) |  256x192   | 0.586 |      0.846      |      0.65       | 0.651 |      0.897      | [ckpt](https://download.openmmlab.com/mmpose/top_down/shufflenetv1/shufflenetv1_coco_256x192-353bc02c_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/shufflenetv1/shufflenetv1_coco_256x192_20200727.log.json) |
| [pose_shufflenetv1](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_shufflenetv1_8xb64-210e_coco-384x288.py) |  384x288   | 0.622 |      0.859      |      0.683      | 0.684 |      0.903      | [ckpt](https://download.openmmlab.com/mmpose/top_down/shufflenetv1/shufflenetv1_coco_384x288-b2930b24_20200804.pth) | [log](https://download.openmmlab.com/mmpose/top_down/shufflenetv1/shufflenetv1_coco_384x288_20200804.log.json) |

<br/>

### Body 2d Keypoint + Resnet + Fp16 + Coco on Coco

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

<!-- [OTHERS] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1710.03740">FP16 (ArXiv'2017)</a></summary>

```bibtex
@article{micikevicius2017mixed,
  title={Mixed precision training},
  author={Micikevicius, Paulius and Narang, Sharan and Alben, Jonah and Diamos, Gregory and Elsen, Erich and Garcia, David and Ginsburg, Boris and Houston, Michael and Kuchaiev, Oleksii and Venkatesh, Ganesh and others},
  journal={arXiv preprint arXiv:1710.03740},
  year={2017}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_resnet_50_fp16](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_fp16-8xb64-210e_coco-256x192.py) |  256x192   | 0.716 |      0.897      |      0.793      | 0.772 |      0.935      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192_fp16_dynamic-6edb79f3_20210430.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192_fp16_dynamic_20210430.log.json) |

<br/>

### Body 2d Keypoint + Litehrnet + Coco on Coco

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2104.06403">LiteHRNet (CVPR'2021)</a></summary>

```bibtex
@inproceedings{Yulitehrnet21,
  title={Lite-HRNet: A Lightweight High-Resolution Network},
  author={Yu, Changqian and Xiao, Bin and Gao, Changxin and Yuan, Lu and Zhang, Lei and Sang, Nong and Wang, Jingdong},
  booktitle={CVPR},
  year={2021}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [LiteHRNet-18](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_litehrnet-18_8xb64-210e_coco-256x192.py) |  256x192   | 0.642 |      0.867      |      0.719      | 0.705 |      0.911      | [ckpt](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet18_coco_256x192-6bace359_20211230.pth) | [log](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet18_coco_256x192_20211230.log.json) |
| [LiteHRNet-18](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_litehrnet-18_8xb32-210e_coco-384x288.py) |  384x288   | 0.676 |      0.876      |      0.746      | 0.735 |      0.919      | [ckpt](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet18_coco_384x288-8d4dac48_20211230.pth) | [log](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet18_coco_384x288_20211230.log.json) |
| [LiteHRNet-30](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_litehrnet-30_8xb64-210e_coco-256x192.py) |  256x192   | 0.676 |      0.88       |      0.756      | 0.736 |      0.922      | [ckpt](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet30_coco_256x192-4176555b_20210626.pth) | [log](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet30_coco_256x192_20210626.log.json) |
| [LiteHRNet-30](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_litehrnet-30_8xb32-210e_coco-384x288.py) |  384x288   |  0.7  |      0.883      |      0.776      | 0.758 |      0.926      | [ckpt](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet30_coco_384x288-a3aef5c4_20210626.pth) | [log](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet30_coco_384x288_20210626.log.json) |

<br/>

### Body 2d Keypoint + RSN + Coco on Coco

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58580-8_27">RSN (ECCV'2020)</a></summary>

```bibtex
@misc{cai2020learning,
    title={Learning Delicate Local Representations for Multi-Person Pose Estimation},
    author={Yuanhao Cai and Zhicheng Wang and Zhengxiong Luo and Binyi Yin and Angang Du and Haoqian Wang and Xinyu Zhou and Erjin Zhou and Xiangyu Zhang and Jian Sun},
    year={2020},
    eprint={2003.04030},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [rsn_18](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_rsn18_8xb32-210e_coco-256x192.py) |  256x192   | 0.703 |      0.887      |      0.78       | 0.77  |      0.927      | [ckpt](https://download.openmmlab.com/mmpose/top_down/rsn/rsn18_coco_256x192-72f4b4a7_20201127.pth) | [log](https://download.openmmlab.com/mmpose/top_down/rsn/rsn18_coco_256x192_20201127.log.json) |
| [rsn_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_rsn50_8xb32-210e_coco-256x192.py) |  256x192   | 0.722 |      0.895      |      0.799      | 0.787 |      0.933      | [ckpt](https://download.openmmlab.com/mmpose/top_down/rsn/rsn50_coco_256x192-72ffe709_20201127.pth) | [log](https://download.openmmlab.com/mmpose/top_down/rsn/rsn50_coco_256x192_20201127.log.json) |
| [2xrsn_50](/configs/body_2d_keypoint/topdown_heatmap/coco/2xtd-hm_rsn50_8xb32-210e_coco-256x192.py) |  256x192   | 0.746 |      0.898      |      0.818      | 0.809 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/top_down/rsn/2xrsn50_coco_256x192-50648f0e_20201127.pth) | [log](https://download.openmmlab.com/mmpose/top_down/rsn/2xrsn50_coco_256x192_20201127.log.json) |
| [3xrsn_50](/configs/body_2d_keypoint/topdown_heatmap/coco/3xtd-hm_rsn50_8xb32-210e_coco-256x192.py) |  256x192   | 0.749 |      0.899      |      0.824      | 0.812 |      0.94       | [ckpt](https://download.openmmlab.com/mmpose/top_down/rsn/3xrsn50_coco_256x192-58f57a68_20201127.pth) | [log](https://download.openmmlab.com/mmpose/top_down/rsn/3xrsn50_coco_256x192_20201127.log.json) |

<br/>

### Body 2d Keypoint + Mobilenetv2 + Coco on Coco

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html">MobilenetV2 (CVPR'2018)</a></summary>

```bibtex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_mobilenetv2](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_mobilenetv2_8xb64-210e_coco-256x192.py) |  256x192   | 0.647 |      0.874      |      0.723      | 0.708 |      0.917      | [ckpt](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_256x192-d1e58e7b_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_256x192_20200727.log.json) |
| [pose_mobilenetv2](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_mobilenetv2_8xb64-210e_coco-384x288.py) |  384x288   | 0.673 |      0.879      |      0.741      | 0.728 |      0.916      | [ckpt](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_384x288-26be4816_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_384x288_20200727.log.json) |

<br/>

### Body 2d Keypoint + Resnetv1d + Coco on Coco

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.html">ResNetV1D (CVPR'2019)</a></summary>

```bibtex
@inproceedings{he2019bag,
  title={Bag of tricks for image classification with convolutional neural networks},
  author={He, Tong and Zhang, Zhi and Zhang, Hang and Zhang, Zhongyue and Xie, Junyuan and Li, Mu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={558--567},
  year={2019}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_resnetv1d_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d50_8xb64-210e_coco-256x192.py) |  256x192   | 0.722 |      0.897      |      0.798      | 0.777 |      0.934      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_coco_256x192-a243b840_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_coco_256x192_20200727.log.json) |
| [pose_resnetv1d_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d50_8xb64-210e_coco-384x288.py) |  384x288   | 0.73  |       0.9       |      0.799      | 0.781 |      0.934      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_coco_384x288-01f3fbb9_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_coco_384x288_20200727.log.json) |
| [pose_resnetv1d_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d101_8xb64-210e_coco-256x192.py) |  256x192   | 0.731 |       0.9       |      0.809      | 0.786 |      0.937      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_coco_256x192-5bd08cab_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_coco_256x192_20200727.log.json) |
| [pose_resnetv1d_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d101_8xb64-210e_coco-384x288.py) |  384x288   | 0.748 |      0.902      |      0.818      | 0.799 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_coco_384x288-5f9e421d_20200730.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_coco_384x288-20200730.log.json) |
| [pose_resnetv1d_152](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d152_8xb32-210e_coco-256x192.py) |  256x192   | 0.737 |      0.902      |      0.813      | 0.791 |      0.938      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_coco_256x192-c4df51dc_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_coco_256x192_20200727.log.json) |
| [pose_resnetv1d_152](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d152_8xb32-210e_coco-384x288.py) |  384x288   | 0.751 |      0.907      |      0.82       | 0.802 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_coco_384x288-626c622d_20200730.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_coco_384x288-20200730.log.json) |

<br/>

### Body 2d Keypoint + Hrnet + Fp16 + Coco on Coco

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

<!-- [OTHERS] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1710.03740">FP16 (ArXiv'2017)</a></summary>

```bibtex
@article{micikevicius2017mixed,
  title={Mixed precision training},
  author={Micikevicius, Paulius and Narang, Sharan and Alben, Jonah and Diamos, Gregory and Elsen, Erich and Garcia, David and Ginsburg, Boris and Houston, Michael and Kuchaiev, Oleksii and Venkatesh, Ganesh and others},
  journal={arXiv preprint arXiv:1710.03740},
  year={2017}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_hrnet_w32_fp16](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_fp16-8xb64-210e_coco-256x192.py) |  256x192   | 0.746 |      0.905      |      0.82       | 0.799 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192_fp16_dynamic-290efc2e_20210430.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192_fp16_dynamic_20210430.log.json) |

<br/>

### Body 2d Keypoint + Seresnet + Coco on Coco

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper">SEResNet (CVPR'2018)</a></summary>

```bibtex
@inproceedings{hu2018squeeze,
  title={Squeeze-and-excitation networks},
  author={Hu, Jie and Shen, Li and Sun, Gang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={7132--7141},
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_seresnet_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_seresnet50_8xb64-210e_coco-256x192.py) |  256x192   | 0.729 |      0.903      |      0.807      | 0.784 |      0.941      | [ckpt](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet50_coco_256x192-25058b66_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet50_coco_256x192_20200727.log.json) |
| [pose_seresnet_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_seresnet50_8xb64-210e_coco-384x288.py) |  384x288   | 0.748 |      0.904      |      0.819      | 0.799 |      0.941      | [ckpt](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet50_coco_384x288-bc0b7680_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet50_coco_384x288_20200727.log.json) |
| [pose_seresnet_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_seresnet101_8xb64-210e_coco-256x192.py) |  256x192   | 0.734 |      0.905      |      0.814      | 0.79  |      0.941      | [ckpt](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet101_coco_256x192-83f29c4d_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet101_coco_256x192_20200727.log.json) |
| [pose_seresnet_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_seresnet101_8xb64-210e_coco-384x288.py) |  384x288   | 0.754 |      0.907      |      0.823      | 0.805 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet101_coco_384x288-48de1709_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet101_coco_384x288_20200727.log.json) |
| [pose_seresnet_152\*](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_seresnet152_8xb32-210e_coco-256x192.py) |  256x192   | 0.73  |      0.899      |      0.81       | 0.787 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet152_coco_256x192-1c628d79_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet152_coco_256x192_20200727.log.json) |
| [pose_seresnet_152\*](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_seresnet152_8xb32-210e_coco-384x288.py) |  384x288   | 0.753 |      0.906      |      0.824      | 0.806 |      0.945      | [ckpt](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet152_coco_384x288-58b23ee8_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet152_coco_384x288_20200727.log.json) |

Note that * means without imagenet pre-training.

<br/>

### Body 2d Keypoint + Resnet + Rle + Coco on Coco

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.html">DeepPose (CVPR'2014)</a></summary>

```bibtex
@inproceedings{toshev2014deeppose,
  title={Deeppose: Human pose estimation via deep neural networks},
  author={Toshev, Alexander and Szegedy, Christian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1653--1660},
  year={2014}
}
```

</details>

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2107.11291">RLE (ICCV'2021)</a></summary>

```bibtex
@inproceedings{li2021human,
  title={Human pose regression with residual log-likelihood estimation},
  author={Li, Jiefeng and Bian, Siyuan and Zeng, Ailing and Wang, Can and Pang, Bo and Liu, Wentao and Lu, Cewu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11025--11034},
  year={2021}
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [deeppose_resnet_50_rle](/configs/body_2d_keypoint/topdown_regression/coco/td-reg_res50_rle-8xb64-210e_coco-256x192.py) |  256x192   | 0.704 |      0.883      |      0.777      | 0.751 |      0.92       | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_coco_256x192_rle-2ea9bb4a_20220616.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_coco_256x192_rle_20220616.log.json) |
| [deeppose_resnet_101_rle](/configs/body_2d_keypoint/topdown_regression/coco/td-reg_res101_rle-8xb64-210e_coco-256x192.py) |  256x192   | 0.722 |      0.894      |      0.794      | 0.768 |      0.93       | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_coco_256x192_rle-16c3d461_20220615.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_coco_256x192_rle_20220615.log.json) |
| [deeppose_resnet_152_rle](/configs/body_2d_keypoint/topdown_regression/coco/td-reg_res152_rle-8xb64-210e_coco-256x192.py) |  256x192   | 0.731 |      0.897      |      0.805      | 0.777 |      0.933      | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_256x192_rle-c05bdccf_20220615.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_256x192_rle_20220615.log.json) |
| [deeppose_resnet_152_rle](/configs/body_2d_keypoint/topdown_regression/coco/td-reg_res152_rle-8xb64-210e_coco-384x288.py) |  384x288   | 0.749 |      0.901      |      0.815      | 0.793 |      0.935      | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_384x288_rle-b77c4c37_20220624.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_384x288_rle_20220624.log.json) |
