<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2107.03332">SimCC (ECCV'2022)</a></summary>

```bibtex
@misc{li20212d,
      title={Is 2D Heatmap Representation Even Necessary for Human Pose Estimation?},
      author={Yanjie Li and Sen Yang and Shoukui Zhang and Zhicheng Wang and Wankou Yang and Shu-Tao Xia and Erjin Zhou},
      year={2021},
      eprint={2107.03332},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

</details>

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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                  | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |    ckpt    |    log    |
| :-------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :--------: | :-------: |
| [simcc_hrnet_w48](<>) |  256x192   | 0.746 |      0.904      |      0.819      | 0.799 |      0.942      | [ckpt](<>) | [log](<>) |
