# Removing the Bias of Integral Pose Regression

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Gu_Removing_the_Bias_of_Integral_Pose_Regression_ICCV_2021_paper.pdf">Debias IPR (ICCV'2021)</a></summary>

```bibtex
@inproceedings{gu2021removing,
    title={Removing the Bias of Integral Pose Regression},
    author={Gu, Kerui and Yang, Linlin and Yao, Angela},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={11067--11076},
    year={2021}
  }
```

</details>

## Abstract

<!-- [ABSTRACT] -->

Heatmap-based detection methods are dominant for 2D human pose estimation even though regression is more intuitive. The introduction of the integral regression method, which, architecture-wise uses an implicit heatmap, brings the two approaches even closer together. This begs the question -- does detection really outperform regression? In this paper, we investigate the difference in supervision between the heatmap-based detection and integral regression, as this is the key remaining difference between the two approaches. In the process, we discover an underlying bias behind integral pose regression that arises from taking the expectation after the softmax function. To counter the bias, we present a compensation method which we find to improve integral regression accuracy on all 2D pose estimation benchmarks. We further propose a simple combined detection and bias-compensated regression method that considerably outperforms state-of-the-art baselines with few added components.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/189810184-159432bb-32a1-403c-8150-e90edce1a5bb.png">
</div>
