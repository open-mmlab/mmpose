# Convolutional pose machines

## Introduction
```
@inproceedings{wei2016convolutional,
  title={Convolutional pose machines},
  author={Wei, Shih-En and Ramakrishna, Varun and Kanade, Takeo and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={4724--4732},
  year={2016}
}
```

## Results and models

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [cpm](/configs/top_down/cpm/coco/cpm_coco_256x192.py)  | 256x192 | 0.623 | 0.859 | 0.704 | 0.686 | 0.903 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/cpm/cpm_coco_256x192-aa4ba095_20200817.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/cpm/cpm_coco_256x192_20200817.log.json) |
