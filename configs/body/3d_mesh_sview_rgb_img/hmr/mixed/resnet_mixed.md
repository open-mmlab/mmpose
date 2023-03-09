<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Kanazawa_End-to-End_Recovery_of_CVPR_2018_paper.html">HMR (CVPR'2018)</a></summary>

```bibtex
@inProceedings{kanazawaHMR18,
  title={End-to-end Recovery of Human Shape and Pose},
  author = {Angjoo Kanazawa
  and Michael J. Black
  and David W. Jacobs
  and Jitendra Malik},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```

</details>

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html">ResNet (CVPR'2016)</a></summary>

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/6682899/">Human3.6M (TPAMI'2014)</a></summary>

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

</details>

Results on Human3.6M with ground-truth bounding box having MPJPE-PA of 52.60 mm on Protocol2

| Arch                                             | Input Size | MPJPE (P1) | MPJPE-PA (P1) | MPJPE (P2) | MPJPE-PA (P2) |                       ckpt                       |                       log                        |
| :----------------------------------------------- | :--------: | :--------: | :-----------: | :--------: | :-----------: | :----------------------------------------------: | :----------------------------------------------: |
| [hmr_resnet_50](/configs/body/3d_mesh_sview_rgb_img/hmr/mixed/res50_mixed_224x224.py) |  224x224   |   80.75    |     55.08     |   80.35    |     52.60     | [ckpt](https://download.openmmlab.com/mmpose/mesh/hmr/hmr_mesh_224x224-c21e8229_20201015.pth) | [log](https://download.openmmlab.com/mmpose/mesh/hmr/hmr_mesh_224x224_20201015.log.json) |
