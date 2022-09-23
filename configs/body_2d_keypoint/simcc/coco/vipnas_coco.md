<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2107.03332">SimCC (ECCV'2022)</a></summary>

```bibtex
@misc{https://doi.org/10.48550/arxiv.2107.03332,
  title={SimCC: a Simple Coordinate Classification Perspective for Human Pose Estimation},
  author={Li, Yanjie and Yang, Sen and Liu, Peidong and Zhang, Shoukui and Wang, Yunxiao and Wang, Zhicheng and Yang, Wankou and Xia, Shu-Tao},
  year={2021}
}
```

</details>

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2105.10154">ViPNAS (CVPR'2021)</a></summary>

```bibtex
@article{xu2021vipnas,
  title={ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search},
  author={Xu, Lumin and Guan, Yingda and Jin, Sheng and Liu, Wentao and Qian, Chen and Luo, Ping and Ouyang, Wanli and Wang, Xiaogang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2021}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48">COCO (ECCV'2014)</a></summary>

```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

</details>

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [simcc_S-ViPNAS-MobileNetV3](/configs/body_2d_keypoint/simcc/coco/simcc_vipnas-mbv3_8xb64-210e_coco-256x192.py) |  256x192   | 0.695 |      0.883      |      0.772      | 0.755 |      0.927      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/coco/simcc_vipnas-mbv3_8xb64-210e_coco-256x192-719f3489_20220922.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/coco/simcc_vipnas-mbv3_8xb64-210e_coco-256x192-719f3489_20220922.log.json) |
