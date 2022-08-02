<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1611.05424">Associative Embedding (NIPS'2017)</a></summary>

```bibtex
@inproceedings{newell2017associative,
  title={Associative embedding: End-to-end learning for joint detection and grouping},
  author={Newell, Alejandro and Huang, Zhiao and Deng, Jia},
  booktitle={Advances in neural information processing systems},
  pages={2277--2287},
  year={2017}
}
```

</details>

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1611.05424">HourglassAENet (NIPS'2017)</a></summary>

```bibtex
@inproceedings{newell2017associative,
  title={Associative embedding: End-to-end learning for joint detection and grouping},
  author={Newell, Alejandro and Huang, Zhiao and Deng, Jia},
  booktitle={Advances in neural information processing systems},
  pages={2277--2287},
  year={2017}
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

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hourglass_ae](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hourglass_ae_coco_512x512.py)  | 512x512 | 0.613 | 0.833 | 0.667 | 0.659 | 0.850 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hourglass_ae/hourglass_ae_coco_512x512-90af499f_20210920.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hourglass_ae/hourglass_ae_coco_512x512_20210920.log.json) |

Results on COCO val2017 with multi-scale test. 3 default scales (\[2, 1, 0.5\]) are used

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hourglass_ae](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hourglass_ae_coco_512x512.py)  | 512x512 | 0.667 | 0.855 | 0.723 | 0.707 | 0.877 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hourglass_ae/hourglass_ae_coco_512x512-90af499f_20210920.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hourglass_ae/hourglass_ae_coco_512x512_20210920.log.json) |
