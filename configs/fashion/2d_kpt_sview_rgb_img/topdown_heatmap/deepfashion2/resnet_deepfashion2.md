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
| short_sleeved_shirt   | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_short_sleeved_shirt_256x192.py) |  256x256   |  0.988  | 0.703 | 10.2 | [ckpt](https://drive.google.com/file/d/1waYaWYPXum9Lncv4qmd0H7lt8Abgb0Ry/view?usp=share_link) | [log](https://drive.google.com/file/d/1RCsyvHDkAiA_SMwLoQ9YUvgq8jrCcmUm/view?usp=share_link) |
| long_sleeved_shirt    | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_long_sleeved_shirt_256x192.py) |  256x256   |  0.973  | 0.587 | 16.5 | [ckpt](https://drive.google.com/file/d/1KG7SfrNoa7-g-CwmPxUi99qS9peHvdVv/view?usp=share_link) | [log](https://drive.google.com/file/d/1RCsyvHDkAiA_SMwLoQ9YUvgq8jrCcmUm/view?usp=share_link) |
| short_sleeved_outwear | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_short_sleeved_outwear_256x192.py) |  256x256   |  0.966  | 0.408 | 24.0 | [ckpt](https://drive.google.com/file/d/1t9QtnpWIq6uajXgApjj90zWmdoP-MLox/view?usp=share_link) | [log](https://drive.google.com/file/d/1uSqi05NSaQUGjthJ-w3V_mii3HnV65qV/view?usp=share_link) |
| long_sleeved_outwear  | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_long_sleeved_outwear_256x192.py) |  256x256   |  0.987  | 0.517 | 18.1 | [ckpt](https://drive.google.com/file/d/1uSqi05NSaQUGjthJ-w3V_mii3HnV65qV/view?usp=share_link) | [log](https://drive.google.com/file/d/1s6mVl5M4FTWF-L-OdWwZTl3oIDEQ9Kbi/view?usp=share_link) |
| vest                  | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_vest_256x192.py) |  256x256   |  0.981  | 0.643 | 12.7 | [ckpt](https://drive.google.com/file/d/1s6mVl5M4FTWF-L-OdWwZTl3oIDEQ9Kbi/view?usp=share_link) | [log](https://drive.google.com/file/d/1s6mVl5M4FTWF-L-OdWwZTl3oIDEQ9Kbi/view?usp=share_link) |
| sling                 | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_sling_256x192.py) |  256x256   |  0.940  | 0.557 | 21.6 | [ckpt](https://drive.google.com/file/d/1s6mVl5M4FTWF-L-OdWwZTl3oIDEQ9Kbi/view?usp=share_link) | [log](https://drive.google.com/file/d/1s6mVl5M4FTWF-L-OdWwZTl3oIDEQ9Kbi/view?usp=share_link) |
| shorts                | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_shorts_256x192.py) |  256x256   |  0.975  | 0.682 | 12.4 | [ckpt](https://drive.google.com/file/d/1nFLP5OPUnbaVGKB38_NEWHCTmVWHu1gp/view?usp=share_link) | [log](https://drive.google.com/file/d/1Uxfz_Uf_SPJlcD8ayRskxtXtkNGc59AG/view?usp=share_link) |
| trousers              | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_trousers_256x192.py) |  256x256   |  0.973  | 0.625 | 14.8 | [ckpt](https://drive.google.com/file/d/1mcSeo2t4Hc5ReBKRPvyFiRrHGNw6n-oX/view?usp=share_link) | [log](https://drive.google.com/file/d/17rBd49cjG9O3NNr5pKydsQqd2DSeXAXT/view?usp=share_link) |
| skirt                 | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_skirt_256x192.py) |  256x256   |  0.952  | 0.653 | 16.6 | [ckpt](https://drive.google.com/file/d/1nyQS09VBW1N4P1V4lXR5SnJWzs5opwWa/view?usp=share_link) | [log](https://drive.google.com/file/d/1qkPuLJVqQBpJ78CMFkbUkiM3kkeXXQNV/view?usp=share_link) |
| short_sleeved_dress   | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_short_sleeved_dress_256x192.py) |  256x256   |  0.980  | 0.603 | 15.6 | [ckpt](https://drive.google.com/file/d/17Hr16XUi0n5zSgmI1-f_aby6oHeXLyXj/view?usp=share_link) | [log](https://drive.google.com/file/d/13k5r5PgnYpMhKoDnjKcDTJOI38c4tQGp/view?usp=share_link) |
| long_sleeved_dress    | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_long_sleeved_dress_256x192.py) |  256x256   |  0.976  | 0.518 | 20.1 | [ckpt](https://drive.google.com/file/d/1zpgTWfShi95GS8Zmjfp-xT7djiGd4LnC/view?usp=share_link) | [log](https://drive.google.com/file/d/1zBsaiV09vgx7T7fYq2DW8dRYhmkRJOhl/view?usp=share_link) |
| vest_dress            | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_vest_dress_256x192.py) |  256x256   |  0.980  | 0.600 | 16.0 | [ckpt](https://drive.google.com/file/d/10egSY-G1ZsD26BmWDLat7snA1boVKL_R/view?usp=share_link) | [log](https://drive.google.com/file/d/1xUzK5D5njvuKWg9pxEa7i5QKpO-WAiN_/view?usp=share_link) |
| sling_dress           | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion2/res50_deepfashion2_sling_dress_256x192.py) |  256x256   |  0.967  | 0.544 | 19.5 | [ckpt](https://drive.google.com/file/d/1gVYjoeD0J0KkZb55FiSdm6sOiLtWHC4r/view?usp=share_link) | [log](https://drive.google.com/file/d/1kMxLVprqR6NNq8RjkGqd7Y7BR9Pq8g_z/view?usp=share_link) |
