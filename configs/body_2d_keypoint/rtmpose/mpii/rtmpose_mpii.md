<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58580-8_27">RTMPose (arXiv'2023)</a></summary>

```bibtex
@misc{https://doi.org/10.48550/arxiv.2303.07399,
  doi = {10.48550/ARXIV.2303.07399},
  url = {https://arxiv.org/abs/2303.07399},
  author = {Jiang, Tao and Lu, Peng and Zhang, Li and Ma, Ningsheng and Han, Rui and Lyu, Chengqi and Li, Yining and Chen, Kai},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}

```

</details>

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

| Arch                                                     | Input Size | Mean / w. flip | Mean@0.1 |                           ckpt                           |                           log                            |
| :------------------------------------------------------- | :--------: | :------------: | :------: | :------------------------------------------------------: | :------------------------------------------------------: |
| [rtmpose-m](/configs/body_2d_keypoint/rtmpose/mpii/rtmpose-m_8xb64-210e_mpii-256x256.py) |  256x256   |     0.907      |  0.348   | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-mpii_pt-aic-coco_210e-256x256-ec4dbec8_20230206.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-mpii_pt-aic-coco_210e-256x256-ec4dbec8_20230206.json) |
