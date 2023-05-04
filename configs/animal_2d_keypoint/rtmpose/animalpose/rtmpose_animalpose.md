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
<summary align="right"><a href="http://openaccess.thecvf.com/content_ICCV_2019/html/Cao_Cross-Domain_Adaptation_for_Animal_Pose_Estimation_ICCV_2019_paper.html">Animal-Pose (ICCV'2019)</a></summary>

```bibtex
@InProceedings{Cao_2019_ICCV,
    author = {Cao, Jinkun and Tang, Hongyang and Fang, Hao-Shu and Shen, Xiaoyong and Lu, Cewu and Tai, Yu-Wing},
    title = {Cross-Domain Adaptation for Animal Pose Estimation},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

</details>

Results on AnimalPose validation set (1117 instances)

| Arch                                                                                                   | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> | ckpt | log |
| :----------------------------------------------------------------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :--: | :-: |
| [rtmpose-t](/configs/animal_2d_keypoint/rtmpose/animalpose/rtmpose-t_8xb64-210e_animalpose-256x256.py) |  256x256   | 0.680 |      0.927      |      0.770      | 0.934 |      0.792      |      |     |
| [rtmpose-s](/configs/animal_2d_keypoint/rtmpose/animalpose/rtmpose-s_8xb64-210e_animalpose-256x256.py) |  256x256   | 0.709 |      0.938      |      0.799      | 0.748 |      0.946      |      |     |
| [rtmpose-m](/configs/animal_2d_keypoint/rtmpose/animalpose/rtmpose-m_8xb64-210e_animalpose-256x256.py) |  256x256   | 0.598 |      0.896      |      0.653      | 0.642 |      0.900      |      |     |
| [rtmpose-l](/configs/animal_2d_keypoint/rtmpose/animalpose/rtmpose-l_8xb64-210e_animalpose-256x256.py) |  256x256   | 0.766 |      0.959      |      0.855      | 0.800 |      0.968      |      |     |
