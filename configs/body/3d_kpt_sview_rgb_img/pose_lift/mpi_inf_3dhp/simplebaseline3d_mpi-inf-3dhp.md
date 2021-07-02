<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_iccv_2017/html/Martinez_A_Simple_yet_ICCV_2017_paper.html">SimpleBaseline3D (ICCV'2017)</a></summary>

```bibtex
@inproceedings{martinez_2017_3dbaseline,
  title={A simple yet effective baseline for 3d human pose estimation},
  author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.},
  booktitle={ICCV},
  year={2017}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/8374605/">MPI-INF-3DHP (3DV'2017)</a></summary>

```bibtex
@inproceedings{mono-3dhp2017,
  author = {Mehta, Dushyant and Rhodin, Helge and Casas, Dan and Fua, Pascal and Sotnychenko, Oleksandr and Xu, Weipeng and Theobalt, Christian},
  title = {Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision},
  booktitle = {3D Vision (3DV), 2017 Fifth International Conference on},
  url = {http://gvv.mpi-inf.mpg.de/3dhp_dataset},
  year = {2017},
  organization={IEEE},
  doi={10.1109/3dv.2017.00064},
}
```

</details>

Results on MPI-INF-3DHP dataset with ground truth 2D detections

| Arch | MPJPE | P-MPJPE | 3DPCK | 3DAUC | ckpt | log |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [simple_baseline_3d_tcn<sup>1</sup>](configs/body/3d_kpt_sview_rgb_img/pose_lift/mpi_inf_3dhp/simplebaseline3d_mpi-inf-3dhp.py) | 84.3 | 53.2 | 85.0 | 52.0 | [ckpt](https://download.openmmlab.com/mmpose/body3d/simplebaseline3d/simplebaseline3d_mpi-inf-3dhp-b75546f6_20210603.pth) | [log](https://download.openmmlab.com/mmpose/body3d/simplebaseline3d/simplebaseline3d_mpi-inf-3dhp_20210603.log.json) |

<sup>1</sup> Differing from the original paper, we didn't apply the `max-norm constraint` because we found this led to a better convergence and performance.
