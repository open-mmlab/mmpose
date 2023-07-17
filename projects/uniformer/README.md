# Pose Estion with UniFormer

This project implements a topdown heatmap based human pose estimator, utilizing the approach outlined in **UniFormer: Unifying Convolution and Self-attention for Visual Recognition** (TPAMI 2023) and **UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning** (ICLR 2022).

<img src="https://raw.githubusercontent.com/Sense-X/UniFormer/main/figures/framework.png" alt><br>

<img src="https://raw.githubusercontent.com/Sense-X/UniFormer/main/figures/dense_adaption.jpg" alt><br>

## Usage

### Preparation

1. Setup Development Environment

- Python 3.7 or higher
- PyTorch 1.6 or higher
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.6.0 or higher
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4 or higher
- [MMDetection](https://github.com/open-mmlab/mmdetection) v3.0.0rc6 or higher
- [MMPose](https://github.com/open-mmlab/mmpose) v1.0.0rc1 or higher

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. **In `uniformer/` root directory**, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

2. Download Pretrained Weights

In order to either running inferences or training on the `uniformer pose estimation` project, you have to download the original Uniformer pretrained weights on ImageNet1k dataset and the weights trained for the downstream pose estimation task. The original ImageNet1k weights are hosted on SenseTime's [huggingface repository](https://huggingface.co/Sense-X/uniformer_image) and the pose estimation weights can be downloaded according to the [official README](https://github.com/Sense-X/UniFormer/tree/main/pose_estimation)

Once you have the weights downloaded, you can move them to the ideal places and update the corresponding paths in the [config files](./configs/) and the main [uniformer.py](./models/uniformer.py)

### Inference

We have provided a [inferencer_demo.py](../../demo/inferencer_demo.py) with which developers can utilize to run quick inference demos. Here is a basic demonstration:

```shell
python demo/inferencer_demo.py $INPUTS \
    --pose2d $CONFIG --pose2d-weights $CHECKPOINT \
    [--show] [--vis-out-dir $VIS_OUT_DIR] [--pred-out-dir $PRED_OUT_DIR]
```

For more information on using the inferencer, please see [this document](https://mmpose.readthedocs.io/en/latest/user_guides/inference.html#out-of-the-box-inferencer).

Here's an example code:

```shell
python demo/inferencer_demo.py ../../tests/data/coco/000000000785.jpg \
    --pose2d configs/yolox-pose_s_8xb32-300e_coco.py \
    --pose2d-weights $PATH_TO_YOUR_UNIFORMER_top_down_256x192_global_base.pth \
    --vis-out-dir vis_results
```

Then you will find the demo result in `vis_results` folder, and it may be similar to this:

<img src="https://github.com/open-mmlab/mmpose/assets/7219519/6f939457-d714-477a-9cc7-27aa98acc4af" height="360px" alt><br>

### Training and Testing

1. Data Preparation

Prepare the COCO dataset according to the [instruction](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#coco).

2. To Train and Test with Single GPU:

```shell
python tools/test.py $CONFIG --auto-scale-lr
```

```shell
python tools/test.py $CONFIG $CHECKPOINT --auto-scale-lr
```

3. To Train and Test with Multiple GPUs:

```shell
bash tools/dist_train.sh $CONFIG $NUM_GPUs --amp
```

```shell
bash tools/dist_test.sh $CONFIG $CHECKPOINT $NUM_GPUs --amp
```

## Results

Results on COCO val2017

|                                Model                                 | Input Size | AP  | AP<sup>50</sup> | AP<sup>75</sup> | AR  | AR<sup>50</sup> |                                Download                                 |
| :------------------------------------------------------------------: | :--------: | :-: | :-------------: | :-------------: | :-: | :-------------: | :---------------------------------------------------------------------: |
| [UniFormer-S](./configs/td-hm_uniformer-s-8xb32-210e_coco-256x192.py) |  256x192   |     |                 |                 |     |                 | [model](https://drive.google.com/file/d/162R0JuTpf3gpLe1IK6oxRoQK7JSj4ylx/view?usp=sharing) \| [log](https://drive.google.com/file/d/15j40u97Db6TA2gMHdn0yFEsDFb5SMBy4/view?usp=sharing) |
| [UniFormer-S](./configs/td-hm_uniformer-s-8xb32-210e_coco-384x288.py) |  384x288   |     |                 |                 |     |                 | [model](https://drive.google.com/file/d/163vuFkpcgVOthC05jCwjGzo78Nr0eikW/view?usp=sharing) \| [log](https://drive.google.com/file/d/15X9M_5cq9RQMgs64Yn9YvV5k5f0zOBHo/view?usp=sharing) |
| [UniFormer-S](./configs/td-hm_uniformer-s-8xb32-210e_coco-448x320.py) |  448x320   |     |                 |                 |     |                 | [model](https://drive.google.com/file/d/165nQRsT58SXJegcttksHwDn46Fme5dGX/view?usp=sharing) \| [log](https://drive.google.com/file/d/15IJjSWp4R5OybMdV2CZEUx_TwXdTMOee/view?usp=sharing) |
| [UniFormer-B](./configs/_base_/td-hm_uniformer-b-8xb32-210e_coco-256x192.py) |  256x192   |     |                 |                 |     |                 | [model](https://drive.google.com/file/d/15tzJaRyEzyWp2mQhpjDbBzuGoyCaJJ-2/view?usp=sharing) \| [log](https://drive.google.com/file/d/15jJyTPcJKj_id0PNdytloqt7yjH2M8UR/view?usp=sharing) |
| [UniFormer-B](./configs/_base_/td-hm_uniformer-b-8xb32-210e_coco-384x288.py) |  384x288   |     |                 |                 |     |                 | [model](https://drive.google.com/file/d/15qtUaOR_C7-vooheJE75mhA9oJQt3gSx/view?usp=sharing) \| [log](https://drive.google.com/file/d/15L1Uxo_uRSMlGnOvWzAzkJLKX6Qh_xNw/view?usp=sharing) |
| [UniFormer-B](./configs/_base_/td-hm_uniformer-b-8xb32-210e_coco-448x320.py) |  448x320   |     |                 |                 |     |                 | [model](https://drive.google.com/file/d/156iNxetiCk8JJz41aFDmFh9cQbCaMk3D/view?usp=sharing) \| [log](https://drive.google.com/file/d/15aRpZc2Tie5gsn3_l-aXto1MrC9wyzMC/view?usp=sharing) |

Note:

1. All the original models are pretrained on ImageNet-1K without Token Labeling and Layer Scale, as mentioned in the [official README](https://github.com/Sense-X/UniFormer/tree/main/pose_estimation) . The official team has confirmed that **Token labeling can largely improve the performance of the downstream tasks**. Developers can utilize the implementation by themselves.
2. The original implementation did not include the **freeze BN in the backbone**. The official team has confirmed that this can improve the performance as well.
3. To avoid running out of memory, developers can use `torch.utils.checkpoint` in the `config.py` by setting `use_checkpoint=True` and `checkpoint_num=[0, 0, 2, 0] # index for using checkpoint in every stage`
4. We warmly welcome any contributions if you can successfully reproduce the results from the paper!

## Citation

If this project benefits your work, please kindly consider citing the original papers:

```bibtex
@misc{li2022uniformer,
      title={UniFormer: Unifying Convolution and Self-attention for Visual Recognition},
      author={Kunchang Li and Yali Wang and Junhao Zhang and Peng Gao and Guanglu Song and Yu Liu and Hongsheng Li and Yu Qiao},
      year={2022},
      eprint={2201.09450},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{li2022uniformer,
      title={UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning},
      author={Kunchang Li and Yali Wang and Peng Gao and Guanglu Song and Yu Liu and Hongsheng Li and Yu Qiao},
      year={2022},
      eprint={2201.04676},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
