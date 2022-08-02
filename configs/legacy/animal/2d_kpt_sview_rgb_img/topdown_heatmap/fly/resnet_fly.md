<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html">SimpleBaseline2D (ECCV'2018)</a></summary>

```bibtex
@inproceedings{xiao2018simple,
  title={Simple baselines for human pose estimation and tracking},
  author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={466--481},
  year={2018}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://www.nature.com/articles/s41592-018-0234-5">Vinegar Fly (Nature Methods'2019)</a></summary>

```bibtex
@article{pereira2019fast,
  title={Fast animal pose estimation using deep neural networks},
  author={Pereira, Talmo D and Aldarondo, Diego E and Willmore, Lindsay and Kislin, Mikhail and Wang, Samuel S-H and Murthy, Mala and Shaevitz, Joshua W},
  journal={Nature methods},
  volume={16},
  number={1},
  pages={117--125},
  year={2019},
  publisher={Nature Publishing Group}
}
```

</details>

Results on Vinegar Fly test set

|  Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :-------- | :--------: | :------: | :------: | :------: |:------: |:------: |
|[pose_resnet_50](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/fly/res50_fly_192x192.py) | 192x192 | 0.996 | 0.910 | 2.00 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_fly_192x192-5d0ee2d9_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_fly_192x192_20210407.log.json) |
|[pose_resnet_101](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/fly/res101_fly_192x192.py) | 192x192 | 0.996 | 0.912 | 1.95 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_fly_192x192-41a7a6cc_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_fly_192x192_20210407.log.json) |
|[pose_resnet_152](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/fly/res152_fly_192x192.py) | 192x192 | 0.997 | 0.917 | 1.78 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_fly_192x192-fcafbd5a_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_fly_192x192_20210407.log.json) |
