<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Contextual_Instance_Decoupling_for_Robust_Multi-Person_Pose_Estimation_CVPR_2022_paper.html">CID (CVPR'2022)</a></summary>

```bibtex
@InProceedings{Wang_2022_CVPR,
    author    = {Wang, Dongkai and Zhang, Shiliang},
    title     = {Contextual Instance Decoupling for Robust Multi-Person Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {11060-11068}
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

Results on COCO val2017 without multi-scale test

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [CID](/configs/body_2d_keypoint/cid/coco/cid_hrnet-w32_8xb20-140e_coco-512x512.py) |  512x512   | 0.704 |      0.894      |      0.775      | 0.753 |      0.928      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/cid/coco/cid_hrnet-w32_8xb20-140e_coco-512x512_42b7e6e6-20230207.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/cid/coco/cid_hrnet-w32_8xb20-140e_coco-512x512_20230207.json) |
| [CID](/configs/body_2d_keypoint/cid/coco/cid_hrnet-w48_8xb20-140e_coco-512x512.py) |  512x512   | 0.715 |      0.900      |      0.782      | 0.765 |      0.935      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/cid/coco/cid_hrnet-w48_8xb20-140e_coco-512x512_a36c3ecf-20230207.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/cid/coco/cid_hrnet-w48_8xb20-140e_coco-512x512_20230207.json) |
