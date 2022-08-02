<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html">MobilenetV2 (CVPR'2018)</a></summary>

```bibtex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Face (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody-Face val set

| Arch  | Input Size | NME | ckpt | log |
| :-------------- | :-----------: | :------: |:------: |:------: |
| [pose_mobilenetv2](/configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_face/mobilenetv2_coco_wholebody_face_256x256.py)  | 256x256 | 0.0612 | [ckpt](https://download.openmmlab.com/mmpose/face/mobilenetv2/mobilenetv2_coco_wholebody_face_256x256-4a3f096e_20210909.pth) | [log](https://download.openmmlab.com/mmpose/face/mobilenetv2/mobilenetv2_coco_wholebody_face_256x256_20210909.log.json) |
