<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2212.07784">RTMDet (ArXiv 2022)</a></summary>

```bibtex
@misc{lyu2022rtmdet,
      title={RTMDet: An Empirical Study of Designing Real-Time Object Detectors},
      author={Chengqi Lyu and Wenwei Zhang and Haian Huang and Yue Zhou and Yudong Wang and Yanyi Liu and Shilong Zhang and Kai Chen},
      year={2022},
      eprint={2212.07784},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

</details>

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Huang_The_Devil_Is_in_the_Details_Delving_Into_Unbiased_Data_CVPR_2020_paper.html">UDP (CVPR'2020)</a></summary>

```bibtex
@InProceedings{Huang_2020_CVPR,
  author = {Huang, Junjie and Zhu, Zheng and Guo, Feng and Huang, Guan},
  title = {The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2108.12617">AP-10K (NeurIPS'2021)</a></summary>

```bibtex
@misc{yu2021ap10k,
      title={AP-10K: A Benchmark for Animal Pose Estimation in the Wild},
      author={Hang Yu and Yufei Xu and Jing Zhang and Wei Zhao and Ziyu Guan and Dacheng Tao},
      year={2021},
      eprint={2108.12617},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

</details>

Results on AP-10K validation set

| Arch                                       | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> |                    ckpt                     |                    log                     |
| :----------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :------------: | :------------: | :-----------------------------------------: | :----------------------------------------: |
| [pose_cspnext_m](/configs/animal_2d_keypoint/topdown_heatmap/ap10k/cspnext-m_udp_8xb64-210e_ap10k-256x256.py) |  256x256   | 0.703 |      0.944      |      0.776      |     0.513      |     0.710      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-m_udp-ap10k_pt-in1k_210e-256x256-1f2d947a_20230123.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-m_udp-ap10k_pt-in1k_210e-256x256-1f2d947a_20230123.json) |
