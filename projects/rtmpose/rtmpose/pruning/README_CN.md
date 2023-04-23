# 使用GroupFisher剪枝RTMPose

# 概述

我们尝试使用 GroupFisher 算法对 RTMPose 模型进行剪枝。具体来说，我们将一个 RTMPose 模型剪枝到与较小的 RTMPose 模型相同的大小，例如将 RTMPose-S 剪枝到 RTMPose-T 的大小。
实验表明，剪枝后的模型比具有相似大小和推理速度的 RTMPose 模型具有更好的性能（AP）。

我们使用能自动确定剪枝结构的 GroupFisher 剪枝算法，将 RTMPose-S 剪枝到 RTMPose-T 的大小。
此外，我们提供了两个版本的剪枝模型，其中一个只使用 coco 数据集，另一个同时使用 coco 和 ai-challenge 数据集。

# 实验结果

| Arch                                                                  | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> | Flops | Params |                   ckpt                    |      log       |
| :-------------------------------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :---: | :----: | :---------------------------------------: | :------------: |
| [rtmpose-s-pruned](./group_fisher_finetune_rtmpose-s_8xb256-420e_coco-256x192.py) |  256x192   | 0.691 |      0.885      |      0.765      | 0.745 |      0.925      | 0.34  |  3.42  | [pruned][rp_sc_p] \| [finetuned][rp_sc_f] | [log][rp_sc_l] |
| [rtmpose-s-aic-coco-pruned](./group_fisher_finetune_rtmpose-s_8xb256-420e_aic-coco-256x192.py) |  256x192   | 0.694 |      0.884      |      0.771      | 0.747 |      0.922      | 0.35  |  3.43  | [pruned][rp_sa_p] \| [finetuned][rp_sa_f] | [log][rp_sa_l] |

## Get Started

我们需要三个步骤来将 GroupFisher 应用于你的模型，包括剪枝（Prune），微调（Finetune），部署（Deploy）。
注意：请使用torch>=1.12，因为我们需要fxtracer来自动解析模型。

### Prune

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 ./tools/dist_train.sh \
  {config_folder}/group_fisher_{normalization_type}_prune_{model_name}.py 8 \
  --work-dir $WORK_DIR
```

在剪枝配置文件中，你需要填写以下参数。

```python
"""
_base_ (str): The path to your pretrained model checkpoint.
pretrained_path (str): The path to your pretrained model checkpoint.

interval (int): Interval between pruning two channels. You should ensure you
    can reach your target pruning ratio when the training ends.
normalization_type (str): GroupFisher uses two methods to normlized the channel
    importance, including ['flops','act']. The former uses flops, while the
    latter uses the memory occupation of activation feature maps.
lr_ratio (float): Ratio to decrease lr rate. As pruning progress is unstable,
    you need to decrease the original lr rate until the pruning training work
    steadly without getting nan.

target_flop_ratio (float): The target flop ratio to prune your model.
input_shape (Tuple): input shape to measure the flops.
"""
```

在剪枝结束后，你将获得一个剪枝模型的 checkpoint，该 checkpoint 的名称为 flops\_{target_flop_ratio}.pth，位于你的 workdir 中。

### Finetune

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 ./tools/dist_train.sh \
   {config_folder}/group_fisher_{normalization_type}_finetune_{model_name}.py 8 \
  --work-dir $WORK_DIR
```

微调时也有一些参数需要你填写。

```python
"""
_base_(str): The path to your pruning config file.
pruned_path (str): The path to the checkpoint of the pruned model.
finetune_lr (float): The lr rate to finetune. Usually, we directly use the lr
    rate of the pretrain.
"""
```

在微调结束后，除了最佳模型的 checkpoint 外，还有一个 fix_subnet.json，它记录了剪枝模型的结构。它将在部署时使用。

### Test

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 ./tools/dist_test.sh \
   {config_folder}/group_fisher_{normalization_type}_finetune_{model_name}.py {checkpoint_path} 8
```

### Deploy

对于剪枝模型，你只需要使用剪枝部署 config 来代替预训练 config 来部署模型的剪枝版本。如果你不熟悉 MMDeploy，请参看[MMDeploy document](https://mmdeploy.readthedocs.io/en/latest/02-how-to-run/convert_model.html)。

```bash
python {mmdeploy}/tools/deploy.py \
    {mmdeploy}/{mmdeploy_config}.py \
    {config_folder}/group_fisher_{normalization_type}_deploy_{model_name}.py \
    {path_to_finetuned_checkpoint}.pth \
    {mmdeploy}/tests/data/tiger.jpeg
```

部署配置文件有如下参数：

```python
"""
_base_ (str): The path to your pretrain config file.
fix_subnet (Union[dict,str]): The dict store the pruning structure or the
    json file including it.
divisor (int): The divisor the make the channel number divisible.
"""
```

divisor 设置十分重要，我们建议你在尝试 \[1,2,4,8,16,32\]，以找到最佳设置。

## Reference

[GroupFisher in MMRazor](https://github.com/open-mmlab/mmrazor/tree/main/configs/pruning/base/group_fisher)

[rp_sa_f]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_finetune_rtmpose-s_8xb256-420e_aic-coco-256x192.pth
[rp_sa_l]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_finetune_rtmpose-s_8xb256-420e_aic-coco-256x192.json
[rp_sa_p]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_prune_rtmpose-s_8xb256-420e_aic-coco-256x192.pth
[rp_sc_f]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_finetune_rtmpose-s_8xb256-420e_coco-256x192.pth
[rp_sc_l]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_finetune_rtmpose-s_8xb256-420e_coco-256x192.json
[rp_sc_p]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_prune_rtmpose-s_8xb256-420e_coco-256x192.pth
