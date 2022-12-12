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
<summary align="right"><a href="https://arxiv.org/pdf/1901.07973.pdf">DeepFashion2 (CVPR'2019)</a></summary>

```bibtex
@article{DeepFashion2,
  author = {Yuying Ge and Ruimao Zhang and Lingyun Wu and Xiaogang Wang and Xiaoou Tang and Ping Luo},
  title={A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images},
  journal={CVPR},
  year={2019}
}
```

</details>

Results on DeepFashion2 val set

| Set                   |                        Arch                         | Input Size | PCK@0.2 |  AUC  | EPE  |                        ckpt                         |                         log                         |
| :-------------------- | :-------------------------------------------------: | :--------: | :-----: | :---: | :--: | :-------------------------------------------------: | :-------------------------------------------------: |
| short_sleeved_shirt   | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_short_sleeved_shirt_256x192.py) |  256x256   |  0.988  | 0.703 | 10.2 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_short_sleeved_shirt_256x192-21e1c5da_20221208.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_short_sleeved_shirt_256x192_20221208.log.json) |
| long_sleeved_shirt    | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_long_sleeved_shirt_256x192.py) |  256x256   |  0.973  | 0.587 | 16.5 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_long_sleeved_shirt_256x192-8679e7e3_20221208.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_long_sleeved_shirt_256x192_20221208.log.json) |
| short_sleeved_outwear | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_short_sleeved_outwear_256x192.py) |  256x256   |  0.966  | 0.408 | 24.0 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_short_sleeved_outwear_256x192-a04c1298_20221208.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_short_sleeved_outwear_256x192_20221208.log.json) |
| long_sleeved_outwear  | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_long_sleeved_outwear_256x192.py) |  256x256   |  0.987  | 0.517 | 18.1 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_long_sleeved_outwear_256x192-31fbaecf_20221208.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_long_sleeved_outwear_256x192_20221208.log.json) |
| vest                  | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_vest_256x192.py) |  256x256   |  0.981  | 0.643 | 12.7 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_vest_256x192-4c48d05c_20221208.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_vest_256x192_20221208.log.json) |
| sling                 | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_sling_256x192.py) |  256x256   |  0.940  | 0.557 | 21.6 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_sling_256x192-ebb2b736_20221208.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_sling_256x192_20221208.log.json) |
| shorts                | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_shorts_256x192.py) |  256x256   |  0.975  | 0.682 | 12.4 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_shorts_256x192-9ab23592_20221208.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_shorts_256x192_20221208.log.json) |
| trousers              | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_trousers_256x192.py) |  256x256   |  0.973  | 0.625 | 14.8 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_trousers_256x192-3e632257_20221208.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_trousers_256x192_20221208.log.json) |
| skirt                 | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_skirt_256x192.py) |  256x256   |  0.952  | 0.653 | 16.6 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_skirt_256x192-09573469_20221208.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_skirt_256x192_20221208.log.json) |
| short_sleeved_dress   | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_short_sleeved_dress_256x192.py) |  256x256   |  0.980  | 0.603 | 15.6 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_short_sleeved_dress_256x192-1345b07a_20221208.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_short_sleeved_dress_256x192_20221208.log.json) |
| long_sleeved_dress    | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_long_sleeved_dress_256x192.py) |  256x256   |  0.976  | 0.518 | 20.1 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_long_sleeved_dress_256x192-87bac74e_20221208.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_long_sleeved_dress_256x192_20221208.log.json) |
| vest_dress            | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_vest_dress_256x192.py) |  256x256   |  0.980  | 0.600 | 16.0 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_vest_dress_256x192-fb3fbd6f_20221208.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_vest_dress_256x192_20221208.log.json) |
| sling_dress           | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_sling_dress_256x192.py) |  256x256   |  0.967  | 0.544 | 19.5 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_sling_dress_256x192-8ebae0eb_20221208.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_sling_dress_256x192_20221208.log.json) |
