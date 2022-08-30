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
<summary align="right"><a href="https://arxiv.org/pdf/1904.07399.pdf">AdaptiveWingloss (ICCV'2019)</a></summary>

```bibtex
@inproceedings{wang2019adaptive,
  title={Adaptive wing loss for robust face alignment via heatmap regression},
  author={Wang, Xinyao and Bo, Liefeng and Fuxin, Li},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={6971--6981},
  year={2019}
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

| Arch       | Input Size | NME<sub>*test*</sub> | NME<sub>*pose*</sub> | NME<sub>*illumination*</sub> | NME<sub>*occlusion*</sub> | NME<sub>*blur*</sub> | NME<sub>*makeup*</sub> | NME<sub>*expression*</sub> |    ckpt    |    log    |
| :--------- | :--------: | :------------------: | :------------------: | :--------------------------: | :-----------------------: | :------------------: | :--------------------: | :------------------------: | :--------: | :-------: |
| [pose_hrnetv2_w18_awing](/configs/face_2d_keypoint/topdown_heatmap/wflw/td-hm_hrnetv2-w18_awing-8xb64-60e_wflw-256x256.py) |  256x256   |         4.02         |         6.94         |             3.97             |           4.78            |         4.59         |          3.87          |            4.28            | [ckpt](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_wflw_256x256_awing-5af5055c_20211212.pth) | [log](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_wflw_256x256_awing_20211212.log.json) |
