<!-- [ALGORITHM] -->

```bibtex
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
}
```

<!-- [DATASET] -->

```bibtex
@article{wu2017ai,
  title={Ai challenger: A large-scale dataset for going deeper in image understanding},
  author={Wu, Jiahong and Zheng, He and Zhao, Bo and Li, Yixin and Yan, Baoming and Liang, Rui and Wang, Wenjia and Zhou, Shipei and Lin, Guosen and Fu, Yanwei and others},
  journal={arXiv preprint arXiv:1711.06475},
  year={2017}
}
```

#### Results on AIC validation set without multi-scale test

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [HRNet-w32](/configs/body/2D_Kpt_SView_RGB_Img/AE/aic/hrnet_w32_aic_512x512.py)  | 512x512 | 0.303 | 0.697 | 0.225 | 0.373 | 0.755 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_aic_512x512-77e2a98a_20210131.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_aic_512x512_20210131.log.json) |

#### Results on AIC validation set with multi-scale test. 3 default scales (\[2, 1, 0.5\]) are used

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [HRNet-w32](/configs/body/2D_Kpt_SView_RGB_Img/AE/aic/hrnet_w32_aic_512x512.py)  | 512x512 | 0.318 | 0.717 | 0.246 | 0.379 | 0.764 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_aic_512x512-77e2a98a_20210131.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_aic_512x512_20210131.log.json) |
