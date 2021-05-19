<!-- [ALGORITHM] -->

```bibtex
@inproceedings{wei2016convolutional,
  title={Convolutional pose machines},
  author={Wei, Shih-En and Ramakrishna, Varun and Kanade, Takeo and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={4724--4732},
  year={2016}
}
```

<!-- [DATASET] -->

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt}
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

#### Results on MPII val set

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [cpm](/configs/body/2D_Kpt_SView_RGB_Img/top_down_heatmap/mpii/cpm_mpii_368x368.py) | 368x368 | 0.876 | 0.285 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_mpii_368x368-116e62b8_20200822.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_mpii_368x368_20200822.log.json) |
