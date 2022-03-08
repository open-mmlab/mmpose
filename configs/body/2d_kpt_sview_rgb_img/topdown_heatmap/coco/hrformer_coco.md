<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://proceedings.neurips.cc/paper/2021/hash/3bbfdde8842a5c44a0323518eec97cbe-Abstract.html">HRFormer (NIPS'2021)</a></summary>

```bibtex
@article{yuan2021hrformer,
  title={HRFormer: High-Resolution Vision Transformer for Dense Predict},
  author={Yuan, Yuhui and Fu, Rao and Huang, Lang and Lin, Weihong and Zhang, Chao and Chen, Xilin and Wang, Jingdong},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
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

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrformer_small](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_small_coco_256x192.py)  | 256x192 | 0.737 | 0.899 | 0.810 | 0.792 | 0.938 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_small_coco_256x192-b657896f_20220226.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_small_coco_256x192_20220226.log.json) |
| [pose_hrformer_small](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_small_coco_384x288.py)  | 384x288 | 0.755 | 0.906 | 0.822 | 0.805 | 0.941 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_small_coco_384x288-4b52b078_20220226.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_small_coco_384x288_20220226.log.json) |
| [pose_hrformer_base](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_base_coco_256x192.py)  | 256x192 | 0.753 | 0.907 | 0.821 | 0.806 | 0.943 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_base_coco_256x192-66cee214_20220226.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_base_coco_256x192_20220226.log.json) |
