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
@article{moon2020interhand2,
  title={InterHand2.6M: A dataset and baseline for 3D interacting hand pose estimation from a single RGB image},
  author={Moon, Gyeongsik and Yu, Shoou-I and Wen, He and Shiratori, Takaaki and Lee, Kyoung Mu},
  journal={arXiv preprint arXiv:2008.09309},
  year={2020},
  publisher={Springer}
}
```

#### Results on InterHand2.6M val & test set

|Train Set| Set | Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--- | :--------: | :--------: | :------: | :------: | :------: |:------: |:------: |
|Human_annot|val(M)| [pose_resnet_50](/configs/hand/2D_Kpt_SView_RGB_Img/top_down_heatmap/interhand2d/res50_interhand2d_human_256x256.py) | 256x256 | 0.973 | 0.828 | 5.15 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_human-77b27d1a_20201029.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_human_20201029.log.json) |
|Human_annot|test(H)| [pose_resnet_50](/configs/hand/2D_Kpt_SView_RGB_Img/top_down_heatmap/interhand2d/res50_interhand2d_human_256x256.py) | 256x256 | 0.973 | 0.826 | 5.27 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_human-77b27d1a_20201029.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_human_20201029.log.json) |
|Human_annot|test(M)| [pose_resnet_50](/configs/hand/2D_Kpt_SView_RGB_Img/top_down_heatmap/interhand2d/res50_interhand2d_human_256x256.py) | 256x256 | 0.975 | 0.841 | 4.90 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_human-77b27d1a_20201029.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_human_20201029.log.json) |
|Human_annot|test(H+M)| [pose_resnet_50](/configs/hand/2D_Kpt_SView_RGB_Img/top_down_heatmap/interhand2d/res50_interhand2d_human_256x256.py) | 256x256 | 0.975 | 0.839 | 4.97 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_human-77b27d1a_20201029.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_human_20201029.log.json) |
|Machine_annot|val(M)| [pose_resnet_50](/configs/hand/2D_Kpt_SView_RGB_Img/top_down_heatmap/interhand2d/res50_interhand2d_machine_256x256.py) | 256x256 | 0.970 | 0.824 | 5.39 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_machine-8f3efe9a_20201102.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_machine_20201102.log.json) |
|Machine_annot|test(H)| [pose_resnet_50](/configs/hand/2D_Kpt_SView_RGB_Img/top_down_heatmap/interhand2d/res50_interhand2d_machine_256x256.py) | 256x256 | 0.969 | 0.821 | 5.52 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_machine-8f3efe9a_20201102.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_machine_20201102.log.json) |
|Machine_annot|test(M)| [pose_resnet_50](/configs/hand/2D_Kpt_SView_RGB_Img/top_down_heatmap/interhand2d/res50_interhand2d_machine_256x256.py) | 256x256 | 0.972 | 0.838 | 5.03 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_machine-8f3efe9a_20201102.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_machine_20201102.log.json) |
|Machine_annot|test(H+M)| [pose_resnet_50](/configs/hand/2D_Kpt_SView_RGB_Img/top_down_heatmap/interhand2d/res50_interhand2d_machine_256x256.py) | 256x256 | 0.972 | 0.837 | 5.11 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_machine-8f3efe9a_20201102.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_machine_20201102.log.json) |
|All|val(M)| [pose_resnet_50](/configs/hand/2D_Kpt_SView_RGB_Img/top_down_heatmap/interhand2d/res50_interhand2d_all_256x256.py) | 256x256 | 0.977 | 0.840 | 4.66 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_all-78cc95d4_20201102.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_all_20201102.log.json) |
|All|test(H)| [pose_resnet_50](/configs/hand/2D_Kpt_SView_RGB_Img/top_down_heatmap/interhand2d/res50_interhand2d_all_256x256.py) | 256x256 | 0.979 | 0.839 | 4.65 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_all-78cc95d4_20201102.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_all_20201102.log.json) |
|All|test(M)| [pose_resnet_50](/configs/hand/2D_Kpt_SView_RGB_Img/top_down_heatmap/interhand2d/res50_interhand2d_all_256x256.py) | 256x256 | 0.979 | 0.838 | 4.42 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_all-78cc95d4_20201102.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_all_20201102.log.json) |
|All|test(H+M)| [pose_resnet_50](/configs/hand/2D_Kpt_SView_RGB_Img/top_down_heatmap/interhand2d/res50_interhand2d_all_256x256.py) | 256x256 | 0.979 | 0.851 | 4.46 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_all-78cc95d4_20201102.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_all_20201102.log.json) |
