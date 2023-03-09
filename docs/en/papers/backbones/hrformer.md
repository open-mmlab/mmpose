# HRFormer: High-Resolution Vision Transformer for Dense Predict

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

## Abstract

<!-- [ABSTRACT] -->

We present a High-Resolution Transformer (HRFormer) that learns high-resolution representations for dense
prediction tasks, in contrast to the original Vision Transformer that produces low-resolution representations
and has high memory and computational cost. We take advantage of the multi-resolution parallel design
introduced in high-resolution convolutional networks (HRNet), along with local-window self-attention
that performs self-attention over small non-overlapping image windows, for improving the memory and
computation efficiency. In addition, we introduce a convolution into the FFN to exchange information
across the disconnected image windows. We demonstrate the effectiveness of the HighResolution Transformer
on both human pose estimation and semantic segmentation tasks, e.g., HRFormer outperforms Swin
transformer by 1.3 AP on COCO pose estimation with 50% fewer parameters and 30% fewer FLOPs.
Code is available at: https://github.com/HRNet/HRFormer

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28900607/155838218-a6aa12b5-5855-45ed-922e-1cfe74f63027.png">
</div>
