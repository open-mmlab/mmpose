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
<summary align="right"><a href="http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm">300WLP (IEEE'2017)</a></summary>

```bibtex
@article{zhu2017face,
  title={Face alignment in full pose range: A 3d total solution},
  author={Zhu, Xiangyu and Liu, Xiaoming and Lei, Zhen and Li, Stan Z},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2017},
  publisher={IEEE}
}
```

</details>

Results on 300W-LP dataset

The model is trained on 300W-LP train.

| Arch                                               | Input Size | NME<sub>*full*</sub> | NME<sub>*test*</sub> |                        ckpt                        |                        log                         |
| :------------------------------------------------- | :--------: | :------------------: | :------------------: | :------------------------------------------------: | :------------------------------------------------: |
| [pose_hrnetv2_w18](/configs/face_2d_keypoint/topdown_heatmap/300wlp/td-hm_hrnetv2-w18_8xb64-60e_300wlp-256x256.py) |  256x256   |        0.0413        |       0.04125        | [ckpt](https://download.openmmlab.com/mmpose/v1/face_2d_keypoint/topdown_heatmap/300wlp/hrnetv2_w18_300wlp_256x256-fb433d21_20230922.pth) | [log](https://download.openmmlab.com/mmpose/v1/face_2d_keypoint/topdown_heatmap/300wlp/hrnetv2_w18_300wlp_256x256-fb433d21_20230922.json) |
