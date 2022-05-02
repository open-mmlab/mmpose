# Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2102.12122">PVT (ICCV'2021)</a></summary>

```bibtex
@inproceedings{wang2021pyramid,
  title={Pyramid vision transformer: A versatile backbone for dense prediction without convolutions},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Fan, Deng-Ping and Song, Kaitao and Liang, Ding and Lu, Tong and Luo, Ping and Shao, Ling},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={568--578},
  year={2021}
}
```

</details>

## Abstract

<!-- [ABSTRACT] -->

Although using convolutional neural networks (CNNs) as backbones achieves great
successes in computer vision, this work investigates a simple backbone network
useful for many dense prediction tasks without convolutions. Unlike the
recently-proposed Transformer model (e.g., ViT) that is specially designed for
image classification, we propose Pyramid Vision Transformer~(PVT), which overcomes
the difficulties of porting Transformer to various dense prediction tasks.
PVT has several merits compared to prior arts. (1) Different from ViT that
typically has low-resolution outputs and high computational and memory cost,
PVT can be not only trained on dense partitions of the image to achieve high
output resolution, which is important for dense predictions but also using a
progressive shrinking pyramid to reduce computations of large feature maps.
(2) PVT inherits the advantages from both CNN and Transformer, making it a
unified backbone in various vision tasks without convolutions by simply replacing
CNN backbones. (3) We validate PVT by conducting extensive experiments, showing
that it boosts the performance of many downstream tasks, e.g., object detection,
semantic, and instance segmentation. For example, with a comparable number of
parameters, RetinaNet+PVT achieves 40.4 AP on the COCO dataset, surpassing
RetinNet+ResNet50 (36.3 AP) by 4.1 absolute AP. We hope PVT could serve as an
alternative and useful backbone for pixel-level predictions and facilitate future
researches. Code is available at https://github.com/whai362/PVT .

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28900607/166134407-d455c57e-4ec3-4951-be9e-730cb9d9c213.png">
</div>
