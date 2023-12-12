<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2204.06806">YOLO-Pose (CVPRW'2022)</a></summary>

```bibtex
@inproceedings{maji2022yolo,
  title={Yolo-pose: Enhancing yolo for multi person pose estimation using object keypoint similarity loss},
  author={Maji, Debapriya and Nagori, Soyeb and Mathew, Manu and Poddar, Deepak},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2637--2646},
  year={2022}
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
