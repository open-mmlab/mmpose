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

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Feng_Wing_Loss_for_CVPR_2018_paper.html">Wingloss (CVPR'2018)</a></summary>

```bibtex
@inproceedings{feng2018wing,
  title={Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks},
  author={Feng, Zhen-Hua and Kittler, Josef and Awais, Muhammad and Huber, Patrik and Wu, Xiao-Jun},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2018 IEEE Conference on},
  year={2018},
  pages ={2235-2245},
  organization={IEEE}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Look_at_Boundary_CVPR_2018_paper.html">WFLW (CVPR'2018)</a></summary>

```bibtex
@inproceedings{wu2018look,
  title={Look at boundary: A boundary-aware face alignment algorithm},
  author={Wu, Wayne and Qian, Chen and Yang, Shuo and Wang, Quan and Cai, Yici and Zhou, Qiang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2129--2138},
  year={2018}
}
```

</details>

Results on WFLW dataset

The model is trained on WFLW train.

| Arch  | Input Size | NME<sub>*test*</sub> | NME<sub>*pose*</sub> | NME<sub>*illumination*</sub> | NME<sub>*occlusion*</sub> | NME<sub>*blur*</sub> | NME<sub>*makeup*</sub> | NME<sub>*expression*</sub> | ckpt | log |
| :-----| :--------: | :------------------: | :------------------: |:---------------------------: |:------------------------: | :------------------: | :--------------: |:-------------------------: |:---: | :---: |
| [deeppose_res50_wingloss](/configs/face/2d_kpt_sview_rgb_img/deeppose/wflw/res50_wflw_256x256_wingloss.py)  | 256x256 | 4.64 | 8.25 | 4.59 | 5.56 | 5.26 | 4.59 | 5.07 | [ckpt](https://download.openmmlab.com/mmpose/face/deeppose/deeppose_res50_wflw_256x256_wingloss-f82a5e53_20210303.pth) | [log](https://download.openmmlab.com/mmpose/face/deeppose/deeppose_res50_wflw_256x256_wingloss_20210303.log.json) |
