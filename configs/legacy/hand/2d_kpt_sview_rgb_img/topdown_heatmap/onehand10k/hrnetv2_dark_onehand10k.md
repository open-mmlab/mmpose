<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/9052469/">HRNetv2 (TPAMI'2019)</a></summary>

```bibtex
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
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
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/8529221/">OneHand10K (TCSVT'2019)</a></summary>

```bibtex
@article{wang2018mask,
  title={Mask-pose cascaded cnn for 2d hand pose estimation from single color image},
  author={Wang, Yangang and Peng, Cong and Liu, Yebin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={29},
  number={11},
  pages={3258--3268},
  year={2018},
  publisher={IEEE}
}
```

</details>

Results on OneHand10K val set

| Arch                                                       | Input Size | PCK@0.2 |  AUC  |  EPE  |                            ckpt                            |                            log                            |
| :--------------------------------------------------------- | :--------: | :-----: | :---: | :---: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| [pose_hrnetv2_w18_dark](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/hrnetv2_w18_onehand10k_256x256_dark.py) |  256x256   |  0.990  | 0.573 | 23.84 | [ckpt](https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_onehand10k_256x256_dark-a2f80c64_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_onehand10k_256x256_dark_20210330.log.json) |
