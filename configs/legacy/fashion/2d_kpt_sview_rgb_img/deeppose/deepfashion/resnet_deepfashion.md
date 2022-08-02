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
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2016/html/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.html">DeepFashion (CVPR'2016)</a></summary>

```bibtex
@inproceedings{liuLQWTcvpr16DeepFashion,
 author = {Liu, Ziwei and Luo, Ping and Qiu, Shi and Wang, Xiaogang and Tang, Xiaoou},
 title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
 booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 month = {June},
 year = {2016}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-319-46475-6_15">DeepFashion (ECCV'2016)</a></summary>

```bibtex
@inproceedings{liuYLWTeccv16FashionLandmark,
 author = {Liu, Ziwei and Yan, Sijie and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
 title = {Fashion Landmark Detection in the Wild},
 booktitle = {European Conference on Computer Vision (ECCV)},
 month = {October},
 year = {2016}
 }
```

</details>

Results on DeepFashion val set

|Set   | Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :---: | :--------: | :------: | :------: | :------: |:------: |:------: |
|upper | [deeppose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/deeppose/deepfashion/res50_deepfashion_upper_256x192.py) | 256x256 | 0.965 | 0.535 | 17.2 | [ckpt](https://download.openmmlab.com/mmpose/fashion/deeppose/deeppose_res50_deepfashion_upper_256x192-497799fb_20210309.pth) | [log](https://download.openmmlab.com/mmpose/fashion/deeppose/deeppose_res50_deepfashion_upper_256x192_20210309.log.json) |
|lower | [deeppose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/deeppose/deepfashion/res50_deepfashion_lower_256x192.py) | 256x256 | 0.971 | 0.678 | 11.8 | [ckpt](https://download.openmmlab.com/mmpose/fashion/deeppose/deeppose_res50_deepfashion_lower_256x192-94e0e653_20210309.pth) | [log](https://download.openmmlab.com/mmpose/fashion/deeppose/deeppose_res50_deepfashion_lower_256x192_20210309.log.json) |
|full  | [deeppose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/deeppose/deepfashion/res50_deepfashion_full_256x192.py)  | 256x256 | 0.983 | 0.602 | 14.0 | [ckpt](https://download.openmmlab.com/mmpose/fashion/deeppose/deeppose_res50_deepfashion_full_256x192-4e0273e2_20210309.pth) | [log](https://download.openmmlab.com/mmpose/fashion/deeppose/deeppose_res50_deepfashion_full_256x192_20210309.log.json) |
