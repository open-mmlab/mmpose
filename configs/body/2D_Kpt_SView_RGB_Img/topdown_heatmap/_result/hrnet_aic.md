
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

#### Results on AIC val set with ground-truth bounding boxes

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnet_w32](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/aic/hrnet_w32_aic_256x192.py) | 256x192 | 0.323 | 0.762 | 0.219 | 0.366 | 0.789 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_aic_256x192-30a4e465_20200826.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_aic_256x192_20200826.log.json) |
