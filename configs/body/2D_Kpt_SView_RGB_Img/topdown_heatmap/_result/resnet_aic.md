<!-- [ALGORITHM] -->

```bibtex
@inproceedings{xiao2018simple,
  title={Simple baselines for human pose estimation and tracking},
  author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={466--481},
  year={2018}
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
| [pose_resnet_101](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/aic/res101_aic_256x192.py) | 256x192 | 0.294 | 0.736 | 0.174 | 0.337 | 0.763 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_aic_256x192-79b35445_20200826.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_aic_256x192_20200826.log.json) |
