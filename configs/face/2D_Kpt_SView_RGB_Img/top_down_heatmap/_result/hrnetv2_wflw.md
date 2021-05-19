<!-- [ALGORITHM] -->

```bibtex
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI}
  year={2019}
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
| [pose_hrnetv2_w18](/configs/face/2D_Kpt_SView_RGB_Img/top_down_heatmap/wflw/hrnetv2_w18_wflw_256x256.py)  | 256x256 | 4.06 | 6.98 | 3.99 | 4.83 | 4.59 | 3.92 | 4.33 | [ckpt](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_wflw_256x256-2bf032a6_20210125.pth) | [log](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_wflw_256x256_20210125.log.json) |
