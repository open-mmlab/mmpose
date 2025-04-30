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

| Arch                                              | Input Size | NME<sub>*full*</sub> | NME<sub>*frontal*</sub> |                       ckpt                        |                        log                        |
| :------------------------------------------------ | :--------: | :------------------: | :---------------------: | :-----------------------------------------------: | :-----------------------------------------------: |
| [pose_hrnetv2_w18](/configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py) |  256x256   |         1.41         |          1.27           | [ckpt](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth) | [log](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256_20210125.log.json) |
