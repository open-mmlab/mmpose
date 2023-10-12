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

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2107.08430">YOLOX</a></summary>

```bibtex
@article{ge2021yolox,
  title={Yolox: Exceeding yolo series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
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

Results on COCO val2017

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [yoloxpose_tiny](/configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_tiny_4xb64-300e_coco-416.py) |  416x416   | 0.526 |      0.793      |      0.556      | 0.571 |      0.833      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_tiny_4xb64-300e_coco-416-76eb44ca_20230829.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_tiny_4xb64-300e_coco-416-20230829.json) |
| [yoloxpose_s](/configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_s_8xb32-300e_coco-640.py) |  640x640   | 0.641 |      0.872      |      0.702      | 0.682 |      0.902      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_s_8xb32-300e_coco-640-56c79c1f_20230829.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_s_8xb32-300e_coco-640-20230829.json) |
| [yoloxpose_m](/configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_m_8xb32-300e_coco-640.py) |  640x640   | 0.695 |      0.899      |      0.766      | 0.733 |      0.926      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_m_8xb32-300e_coco-640-84e9a538_20230829.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_m_8xb32-300e_coco-640-20230829.json) |
| [yoloxpose_l](/configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_l_8xb32-300e_coco-640.py) |  640x640   | 0.712 |      0.901      |      0.782      | 0.749 |      0.926      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_l_8xb32-300e_coco-640-de0f8dee_20230829.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_l_8xb32-300e_coco-640-20230829.json) |
