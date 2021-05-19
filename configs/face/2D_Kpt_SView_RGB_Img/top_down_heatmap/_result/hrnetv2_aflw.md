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
@inproceedings{koestinger2011annotated,
  title={Annotated facial landmarks in the wild: A large-scale, real-world database for facial landmark localization},
  author={Koestinger, Martin and Wohlhart, Paul and Roth, Peter M and Bischof, Horst},
  booktitle={2011 IEEE international conference on computer vision workshops (ICCV workshops)},
  pages={2144--2151},
  year={2011},
  organization={IEEE}
}
```

#### Results on AFLW dataset

The model is trained on AFLW train and evaluated on AFLW full and frontal.

| Arch  | Input Size | NME<sub>*full*</sub> | NME<sub>*frontal*</sub>  | ckpt | log |
| :-------------- | :-----------: | :------: | :------: |:------: |:------: |
| [pose_hrnetv2_w18](/configs/face/2D_Kpt_SView_RGB_Img/top_down_heatmap/aflw/hrnetv2_w18_aflw_256x256.py)  | 256x256 | 1.41 | 1.27 | [ckpt](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth) | [log](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256_20210125.log.json) |
