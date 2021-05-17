<!-- [ALGORITHM] -->

```bibtex
@inproceedings{xiao2018simple,
  title={Simple baselines for human pose estimation and tracking},
  author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={466--481},
  year={2018}
}
```

<!-- [DATASET] -->

```bibtex
@inproceedings{liuLQWTcvpr16DeepFashion,
 author = {Liu, Ziwei and Luo, Ping and Qiu, Shi and Wang, Xiaogang and Tang, Xiaoou},
 title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
 booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 month = {June},
 year = {2016}
}
```

<!-- [DATASET] -->

```bibtex
@inproceedings{liuYLWTeccv16FashionLandmark,
 author = {Liu, Ziwei and Yan, Sijie and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
 title = {Fashion Landmark Detection in the Wild},
 booktitle = {European Conference on Computer Vision (ECCV)},
 month = {October},
 year = {2016}
 }
```

#### Results on DeepFashion val set

|Set   | Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :---: | :--------: | :------: | :------: | :------: |:------: |:------: |
|upper | [pose_resnet_50](/configs/fashion/2D_Kpt_SView_RGB_Img/topdown_heatmap/deepfashion/res50_deepfashion_upper_256x192.py) | 256x256 | 0.954 | 0.578 | 16.8 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_upper_256x192-41794f03_20210124.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_upper_256x192_20210124.log.json) |
|lower | [pose_resnet_50](/configs/fashion/2D_Kpt_SView_RGB_Img/topdown_heatmap/deepfashion/res50_deepfashion_lower_256x192.py) | 256x256 | 0.965 | 0.744 | 10.5 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_lower_256x192-1292a839_20210124.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_lower_256x192_20210124.log.json) |
|full  | [pose_resnet_50](/configs/fashion/2D_Kpt_SView_RGB_Img/topdown_heatmap/deepfashion/res50_deepfashion_full_256x192.py)  | 256x256 | 0.977 | 0.664 | 12.7 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_full_256x192-0dbd6e42_20210124.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_full_256x192_20210124.log.json) |
