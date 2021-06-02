<!-- [ALGORITHM] -->

<details>
<summary align="right">SimpleBaseline2D (ECCV'2018)</summary>

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
<summary align="right">ResNet (CVPR'2016)</summary>

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
<summary align="right">InterHand2.6M (ECCV'2020)</summary>

```bibtex
@InProceedings{Moon_2020_ECCV_InterHand2.6M,
author = {Moon, Gyeongsik and Yu, Shoou-I and Wen, He and Shiratori, Takaaki and Lee, Kyoung Mu},
title = {InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image},
booktitle = {European Conference on Computer Vision (ECCV)},
year = {2020}
}
```

</details>

Results on InterHand2.6M val & test set

|Train Set| Set | Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--- | :--------: | :--------: | :------: | :------: | :------: |:------: |:------: |
|Human_annot|val(M)| [pose_resnet_50](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/res50_interhand2d_human_256x256.py) | 256x256 | 0.973 | 0.828 | 5.15 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_human-77b27d1a_20201029.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_human_20201029.log.json) |
|Human_annot|test(H)| [pose_resnet_50](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/res50_interhand2d_human_256x256.py) | 256x256 | 0.973 | 0.826 | 5.27 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_human-77b27d1a_20201029.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_human_20201029.log.json) |
|Human_annot|test(M)| [pose_resnet_50](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/res50_interhand2d_human_256x256.py) | 256x256 | 0.975 | 0.841 | 4.90 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_human-77b27d1a_20201029.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_human_20201029.log.json) |
|Human_annot|test(H+M)| [pose_resnet_50](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/res50_interhand2d_human_256x256.py) | 256x256 | 0.975 | 0.839 | 4.97 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_human-77b27d1a_20201029.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_human_20201029.log.json) |
|Machine_annot|val(M)| [pose_resnet_50](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/res50_interhand2d_machine_256x256.py) | 256x256 | 0.970 | 0.824 | 5.39 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_machine-8f3efe9a_20201102.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_machine_20201102.log.json) |
|Machine_annot|test(H)| [pose_resnet_50](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/res50_interhand2d_machine_256x256.py) | 256x256 | 0.969 | 0.821 | 5.52 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_machine-8f3efe9a_20201102.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_machine_20201102.log.json) |
|Machine_annot|test(M)| [pose_resnet_50](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/res50_interhand2d_machine_256x256.py) | 256x256 | 0.972 | 0.838 | 5.03 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_machine-8f3efe9a_20201102.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_machine_20201102.log.json) |
|Machine_annot|test(H+M)| [pose_resnet_50](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/res50_interhand2d_machine_256x256.py) | 256x256 | 0.972 | 0.837 | 5.11 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_machine-8f3efe9a_20201102.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_machine_20201102.log.json) |
|All|val(M)| [pose_resnet_50](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/res50_interhand2d_all_256x256.py) | 256x256 | 0.977 | 0.840 | 4.66 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_all-78cc95d4_20201102.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_all_20201102.log.json) |
|All|test(H)| [pose_resnet_50](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/res50_interhand2d_all_256x256.py) | 256x256 | 0.979 | 0.839 | 4.65 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_all-78cc95d4_20201102.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_all_20201102.log.json) |
|All|test(M)| [pose_resnet_50](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/res50_interhand2d_all_256x256.py) | 256x256 | 0.979 | 0.838 | 4.42 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_all-78cc95d4_20201102.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_all_20201102.log.json) |
|All|test(H+M)| [pose_resnet_50](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/res50_interhand2d_all_256x256.py) | 256x256 | 0.979 | 0.851 | 4.46 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_all-78cc95d4_20201102.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_all_20201102.log.json) |
