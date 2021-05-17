<!-- [BACKBONE] -->

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

<!-- [DATASET] -->

```bibtex
@article{h36m_pami,
  author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu,  Cristian},
  title = {Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  publisher = {IEEE Computer Society},
  volume = {36},
  number = {7},
  pages = {1325-1339},
  month = {jul},
  year = {2014}
}
```

#### Results on Human3.6M with ground-truth bounding box having MPJPE-PA of 52.60 mm on Protocol2

| Arch  | Input Size | MPJPE (P1)| MPJPE-PA (P1) | MPJPE (P2) | MPJPE-PA (P2) | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |
| [hmr_resnet_50](/configs/body/3D_Mesh_SView_RGB_Img/hmr/hmr_res50_224x224.py)  | 224x224 | 80.75 | 55.08 | 80.35 | 52.60 | [ckpt](https://download.openmmlab.com/mmpose/mesh/hmr/hmr_mesh_224x224-c21e8229_20201015.pth) | [log](https://download.openmmlab.com/mmpose/mesh/hmr/hmr_mesh_224x224_20201015.log.json) |
