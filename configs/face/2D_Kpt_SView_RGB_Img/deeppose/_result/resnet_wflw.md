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
| [deeppose_res50](/configs/face/2D_Kpt_SView_RGB_Img/deeppose/wflw/deeppose_res50_wflw_256x256.py)  | 256x256 | 4.85 | 8.50 | 4.81 | 5.69 | 5.45 | 4.82 | 5.20 | [ckpt](https://download.openmmlab.com/mmpose/face/deeppose/deeppose_res50_wflw_256x256-92d0ba7f_20210303.pth) | [log](https://download.openmmlab.com/mmpose/face/deeppose/deeppose_res50_wflw_256x256_20210303.log.json) |
