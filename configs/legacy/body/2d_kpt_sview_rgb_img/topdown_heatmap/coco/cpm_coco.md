<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2016/html/Wei_Convolutional_Pose_Machines_CVPR_2016_paper.html">CPM (CVPR'2016)</a></summary>

```bibtex
@inproceedings{wei2016convolutional,
  title={Convolutional pose machines},
  author={Wei, Shih-En and Ramakrishna, Varun and Kanade, Takeo and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={4724--4732},
  year={2016}
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

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [cpm](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/cpm_coco_256x192.py)  | 256x192 | 0.623 | 0.859 | 0.704 | 0.686 | 0.903 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_coco_256x192-aa4ba095_20200817.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_coco_256x192_20200817.log.json) |
| [cpm](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/cpm_coco_384x288.py)  | 384x288 | 0.650 | 0.864 | 0.725 | 0.708 | 0.905 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_coco_384x288-80feb4bc_20200821.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_coco_384x288_20200821.log.json) |
