<!-- [ALGORITHM] -->

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch                                                | Input Size | Mean / w. flip | Mean@0.1 |                            ckpt                             |                            log                             |
| :-------------------------------------------------- | :--------: | :------------: | :------: | :---------------------------------------------------------: | :--------------------------------------------------------: |
| [rtmpose-m](./rtmpose-m_8xb64-210e_mpii-256x256.py) |  256x256   |     0.907      |  0.348   | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-mpii_pt-aic-coco_210e-256x256-ec4dbec8_20230206.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-mpii_pt-aic-coco_210e-256x256-ec4dbec8_20230206.json) |
