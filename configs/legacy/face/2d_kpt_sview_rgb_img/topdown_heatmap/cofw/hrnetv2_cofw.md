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

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_iccv_2013/html/Burgos-Artizzu_Robust_Face_Landmark_2013_ICCV_paper.html">COFW (ICCV'2013)</a></summary>

```bibtex
@inproceedings{burgos2013robust,
  title={Robust face landmark estimation under occlusion},
  author={Burgos-Artizzu, Xavier P and Perona, Pietro and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={1513--1520},
  year={2013}
}
```

</details>

Results on COFW dataset

The model is trained on COFW train.

| Arch  | Input Size | NME | ckpt | log |
| :-----| :--------: | :----: |:---: | :---: |
| [pose_hrnetv2_w18](/configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/cofw/hrnetv2_w18_cofw_256x256.py)  | 256x256 |  3.40 | [ckpt](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_cofw_256x256-49243ab8_20211019.pth) | [log](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_cofw_256x256_20211019.log.json) |
