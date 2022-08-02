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
|upper | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion/res50_deepfashion_upper_256x192.py) | 256x256 | 0.954 | 0.578 | 16.8 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_upper_256x192-41794f03_20210124.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_upper_256x192_20210124.log.json) |
|lower | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion/res50_deepfashion_lower_256x192.py) | 256x256 | 0.965 | 0.744 | 10.5 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_lower_256x192-1292a839_20210124.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_lower_256x192_20210124.log.json) |
|full  | [pose_resnet_50](/configs/fashion/2d_kpt_sview_rgb_img/topdown_heatmap/deepfashion/res50_deepfashion_full_256x192.py)  | 256x256 | 0.977 | 0.664 | 12.7 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_full_256x192-0dbd6e42_20210124.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_full_256x192_20210124.log.json) |
