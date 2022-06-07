<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Abavisani_Improving_the_Performance_of_Unimodal_Dynamic_Hand-Gesture_Recognition_With_Multimodal_CVPR_2019_paper.html">MTUT (CVPR'2019)</a></summary>

```bibtex
@InProceedings{Abavisani_2019_CVPR,
  author = {Abavisani, Mahdi and Joze, Hamid Reza Vaezi and Patel, Vishal M.},
  title = {Improving the Performance of Unimodal Dynamic Hand-Gesture Recognition With Multimodal Training},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```

</details>

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html">I3D (CVPR'2017)</a></summary>

```bibtex
@InProceedings{Carreira_2017_CVPR,
  author = {Carreira, Joao and Zisserman, Andrew},
  title = {Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {July},
  year = {2017}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content_cvpr_2016/html/Molchanov_Online_Detection_and_CVPR_2016_paper.html">NVGesture (CVPR'2016)</a></summary>

```bibtex
@InProceedings{Molchanov_2016_CVPR,
  author = {Molchanov, Pavlo and Yang, Xiaodong and Gupta, Shalini and Kim, Kihwan and Tyree, Stephen and Kautz, Jan},
  title = {Online Detection and Classification of Dynamic Hand Gestures With Recurrent 3D Convolutional Neural Network},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2016}
}
```

</details>

Results on NVGesture test set

| Arch                                                    | Input Size | fps |   bbox    | AP_rgb | AP_depth |                          ckpt                           |                          log                           |
| :------------------------------------------------------ | :--------: | :-: | :-------: | :----: | :------: | :-----------------------------------------------------: | :----------------------------------------------------: |
| [I3D+MTUT](/configs/hand/gesture_sview_rgbd_vid/mtut/nvgesture/i3d_nvgesture_bbox_112x112_fps15.py)<sup>\*</sup> |  112x112   | 15  | $\\surd$  | 0.725  |  0.730   | [ckpt](https://download.openmmlab.com/mmpose/gesture/mtut/i3d_nvgesture_bbox_112x112_fps15-363b5956_20220530.pth) | [log](https://download.openmmlab.com/mmpose/gesture/mtut/i3d_nvgesture_bbox_112x112_fps15-20220530.log.json) |
| [I3D+MTUT](/configs/hand/gesture_sview_rgbd_vid/mtut/nvgesture/i3d_nvgesture_bbox_224x224_fps30.py) |  224x224   | 30  | $\\surd$  | 0.782  |  0.811   | [ckpt](https://download.openmmlab.com/mmpose/gesture/mtut/i3d_nvgesture_bbox_224x224_fps30-98a8f288_20220530.pthh) | [log](https://download.openmmlab.com/mmpose/gesture/mtut/i3d_nvgesture_bbox_224x224_fps30-20220530.log.json) |
| [I3D+MTUT](/configs/hand/gesture_sview_rgbd_vid/mtut/nvgesture/i3d_nvgesture_224x224_fps30.py) |  224x224   | 30  | $\\times$ | 0.739  |  0.809   | [ckpt](https://download.openmmlab.com/mmpose/gesture/mtut/i3d_nvgesture_224x224_fps30-b7abf574_20220530.pth) | [log](https://download.openmmlab.com/mmpose/gesture/mtut/i3d_nvgesture_224x224_fps30-20220530.log.json) |

<sup>\*</sup>: MTUT supports multi-modal training and uni-modal testing. Model trained with this config can be used to recognize gestures in rgb videos with [inference config](/configs/hand/gesture_sview_rgbd_vid/mtut/nvgesture/i3d_nvgesture_bbox_112x112_fps15_rgb.py).
