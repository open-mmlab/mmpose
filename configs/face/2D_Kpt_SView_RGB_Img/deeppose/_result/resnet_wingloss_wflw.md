<!-- [BACKBONE] -->

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

<!-- [ALGORITHM] -->

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

<!-- [DATASET] -->

```bibtex
@inproceedings{wu2018look,
  title={Look at boundary: A boundary-aware face alignment algorithm},
  author={Wu, Wayne and Qian, Chen and Yang, Shuo and Wang, Quan and Cai, Yici and Zhou, Qiang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2129--2138},
  year={2018}
}
```

#### Results on WFLW dataset

The model is trained on WFLW train.

| Arch  | Input Size | NME<sub>*test*</sub> | NME<sub>*pose*</sub> | NME<sub>*illumination*</sub> | NME<sub>*occlusion*</sub> | NME<sub>*blur*</sub> | NME<sub>*makeup*</sub> | NME<sub>*expression*</sub> | ckpt | log |
| :-----| :--------: | :------------------: | :------------------: |:---------------------------: |:------------------------: | :------------------: | :--------------: |:-------------------------: |:---: | :---: |
| [deeppose_res50_wingloss](/configs/face/2D_Kpt_SView_RGB_Img/deeppose/wflw/deeppose_res50_wflw_256x256_wingloss.py)  | 256x256 | 4.64 | 8.25 | 4.59 | 5.56 | 5.26 | 4.59 | 5.07 | [ckpt](https://download.openmmlab.com/mmpose/face/deeppose/deeppose_res50_wflw_256x256_wingloss-f82a5e53_20210303.pth) | [log](https://download.openmmlab.com/mmpose/face/deeppose/deeppose_res50_wflw_256x256_wingloss_20210303.log.json) |
