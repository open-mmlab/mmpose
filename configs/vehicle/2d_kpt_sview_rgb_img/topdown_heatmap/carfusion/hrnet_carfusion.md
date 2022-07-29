<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html">HRNet (CVPR'2019)</a></summary>

```bibtex
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://www.cs.cmu.edu/~ILIM/publications/PDFs/RVN-CVPR18.pdf">CARFUSION (CVPR'2018)</a></summary>

```bibtex
@InProceedings{Reddy_2018_CVPR,
author = {Dinesh Reddy, N. and Vo, Minh and Narasimhan, Srinivasa G.},
title = {CarFusion: Combining Point Tracking and Part Detection for Dynamic 3D Reconstruction of Vehicles},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```

</details>

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_hrnet_w32](/configs/vehicle/2d_kpt_sview_rgb_img/topdown_heatmap/carfusion/hrnet_w32_carfusion_384x288.py) |  384x288   | 0.760 |      0.906      |      0.829      | 0.810 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_carfusion_384x288-d9f0d786_20200708.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_carfusion_384x288_20200708.log.json) |
