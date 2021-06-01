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
<summary align="right">FreiHand (ICCV'2019)</summary>

```bibtex
@inproceedings{zimmermann2019freihand,
  title={Freihand: A dataset for markerless capture of hand pose and shape from single rgb images},
  author={Zimmermann, Christian and Ceylan, Duygu and Yang, Jimei and Russell, Bryan and Argus, Max and Brox, Thomas},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={813--822},
  year={2019}
}
```

</details>

Results on FreiHand val & test set

| Set | Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :--------: | :------: | :------: | :------: |:------: |:------: |
|val| [pose_resnet_50](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/freihand2d/res50_freihand_224x224.py) | 224x224 | 0.993 | 0.868 | 3.25 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_freihand_224x224-ff0799bc_20200914.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_freihand_224x224_20200914.log.json) |
|test| [pose_resnet_50](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/freihand2d/res50_freihand_224x224.py) | 224x224 | 0.992 | 0.868 | 3.27 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_freihand_224x224-ff0799bc_20200914.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_freihand_224x224_20200914.log.json) |
