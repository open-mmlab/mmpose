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
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/6130513/">AFLW (ICCVW'2011)</a></summary>

```bibtex
@inproceedings{koestinger2011annotated,
  title={Annotated facial landmarks in the wild: A large-scale, real-world database for facial landmark localization},
  author={Koestinger, Martin and Wohlhart, Paul and Roth, Peter M and Bischof, Horst},
  booktitle={2011 IEEE international conference on computer vision workshops (ICCV workshops)},
  pages={2144--2151},
  year={2011},
  organization={IEEE}
}
```

</details>

Results on AFLW dataset

The model is trained on AFLW train and evaluated on AFLW full and frontal.

| Arch  | Input Size | NME<sub>*full*</sub> | NME<sub>*frontal*</sub>  | ckpt | log |
| :-------------- | :-----------: | :------: | :------: |:------: |:------: |
| [pose_hrnetv2_w18_dark](/configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/hrnetv2_w18_aflw_256x256_dark.py)  | 256x256 | 1.34 | 1.20 | [ckpt](https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_aflw_256x256_dark-219606c0_20210125.pth) | [log](https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_aflw_256x256_dark_20210125.log.json) |
