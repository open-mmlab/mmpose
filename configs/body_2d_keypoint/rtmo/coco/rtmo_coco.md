<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2312.07526">RTMO</a></summary>

```bibtex
@misc{lu2023rtmo,
      title={{RTMO}: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation},
      author={Peng Lu and Tao Jiang and Yining Li and Xiangtai Li and Kai Chen and Wenming Yang},
      year={2023},
      eprint={2312.07526},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
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

Results on COCO val2017

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [RTMO-s](/configs/body_2d_keypoint/rtmo/coco/rtmo-s_8xb32-600e_coco-640x640.py) |  640x640   | 0.677 |      0.878      |      0.737      | 0.715 |      0.908      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-s_8xb32-600e_coco-640x640_20231211.json) |
| [RTMO-m](/configs/body_2d_keypoint/rtmo/coco/rtmo-m_16xb16-600e_coco-640x640.py) |  640x640   | 0.709 |      0.890      |      0.778      | 0.747 |      0.920      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-m_16xb16-600e_coco-640x640-6f4e0306_20231211.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-m_16xb16-600e_coco-640x640_20231211.json) |
| [RTMO-l](/configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py) |  640x640   | 0.724 |      0.899      |      0.788      | 0.762 |      0.927      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-600e_coco-640x640_20231211.json) |
