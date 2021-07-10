# A simple yet effective baseline for 3d human pose estimation

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

Simple 3D baseline proposes to break down the task of 3d human pose estimation into 2 stages: (1) Image → 2D pose
(2) 2D pose → 3D pose.

The authors find that “lifting” ground truth 2D joint locations to 3D space is a task that can be solved with a low error rate.
Based on the success of 2d human pose estimation, it directly "lifts" 2d joint locations to 3d space.
