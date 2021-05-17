<!-- [ALGORITHM] -->

```bibtex
@inproceedings{cheng2020higherhrnet,
  title={HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation},
  author={Cheng, Bowen and Xiao, Bin and Wang, Jingdong and Shi, Honghui and Huang, Thomas S and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5386--5395},
  year={2020}
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
| [HigherHRNet-w32](/configs/body/2D_Kpt_SView_RGB_Img/AE/aic/higher_hrnet32_aic_512x512.py)  | 512x512 | 0.315 | 0.710 | 0.243 | 0.379 | 0.757 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_aic_512x512-9a674c33_20210130.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_aic_512x512_20210130.log.json) |

#### Results on AIC validation set with multi-scale test. 3 default scales (\[2, 1, 0.5\]) are used

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [HigherHRNet-w32](/configs/body/2D_Kpt_SView_RGB_Img/AE/aic/higher_hrnet32_aic_512x512.py)  | 512x512 | 0.323 | 0.718 | 0.254 | 0.379 | 0.758 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_aic_512x512-9a674c33_20210130.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_aic_512x512_20210130.log.json) |
