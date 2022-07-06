# Not All Tokens Are Equal: Human-Centric Visual Analysis via Token Clustering Transformer

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content/CVPR2022/html/Zeng_Not_All_Tokens_Are_Equal_Human-Centric_Visual_Analysis_via_Token_CVPR_2022_paper.html">TCFormer (CVPR'2022)</a></summary>

```bibtex
@inproceedings{zeng2022not,
  title={Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer},
  author={Zeng, Wang and Jin, Sheng and Liu, Wentao and Qian, Chen and Luo, Ping and Ouyang, Wanli and Wang, Xiaogang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11101--11111},
  year={2022}
}
```

</details>

## Abstract

<!-- [ABSTRACT] -->

Vision transformers have achieved great successes in many computer vision tasks. Most methods generate vision tokens by splitting an image into a regular and fixed grid and treating each cell as a token. However, not all regions are equally important in human-centric vision tasks, e.g., the human body needs a fine representation with many tokens, while the image background can be modeled by a few tokens. To address this problem, we propose a novel Vision Transformer, called Token Clustering Transformer (TCFormer), which merges tokens by progressive clustering, where the tokens can be merged from different locations with flexible shapes and sizes. The tokens in TCFormer can not only focus on important areas but also adjust the token shapes to fit the semantic concept and adopt a fine resolution for regions containing critical details, which is beneficial to capturing detailed information. Extensive experiments show that TCFormer consistently outperforms its counterparts on different challenging humancentric tasks and datasets, including whole-body pose estimation on COCO-WholeBody and 3D human mesh reconstruction on 3DPW. Code is available at https://github.com/zengwang430521/TCFormer.git.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28900607/175868010-b408e0dc-768c-4fb9-9095-5874fcb42d7f.png">
</div>
