# GroupFisher Pruning for RTMPose

# Description

We try to apply a pruning algorithm to RTMPose models. In detail, we prune a RTMPose model to a smaller size as the same as a smaller RTMPose model, like pruning RTMPose-S to the size of RTMPose-T.
The expriments show that the pruned model have better performance(AP) than the RTMPose model with the similar size and inference speed.

Concretly, we select the RTMPose-S as the base model and prune it to the size of RTMPose-T, and use GroupFisher pruning algorithm which is able to determine the pruning structure automatically.
Furthermore, we provide two version of the pruned models including only using coco and using both of coco and ai-challenge datasets.

# Results and Models

| Arch                                                                  | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> | Flops | Params |                   ckpt                    |      log       |
| :-------------------------------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :---: | :----: | :---------------------------------------: | :------------: |
| [rtmpose-s-pruned](./group_fisher_finetune_rtmpose-s_8xb256-420e_coco-256x192.py) |  256x192   | 0.691 |      0.885      |      0.765      | 0.745 |      0.925      | 0.34  |  3.42  | [pruned][rp_sc_p] \| [finetuned][rp_sc_f] | [log][rp_sc_l] |
| [rtmpose-s-aic-coco-pruned](./group_fisher_finetune_rtmpose-s_8xb256-420e_aic-coco-256x192.py) |  256x192   | 0.694 |      0.884      |      0.771      | 0.747 |      0.922      | 0.35  |  3.43  | [pruned][rp_sa_p] \| [finetuned][rp_sa_f] | [log][rp_sa_l] |

## Get Started

We have three steps to apply GroupFisher to your model, including Prune, Finetune, Deploy.

Note: please use torch>=1.12, as we need fxtracer to parse the models automatically.

### Prune

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 ./tools/dist_train.sh \
  {config_folder}/group_fisher_{normalization_type}_prune_{model_name}.py 8 \
  --work-dir $WORK_DIR
```

In the pruning config file. You have to fill some args as below.

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

After the pruning process, you will get a checkpoint of the pruned model named flops\_{target_flop_ratio}.pth in your workdir.

### Finetune

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 ./tools/dist_train.sh \
   {config_folder}/group_fisher_{normalization_type}_finetune_{model_name}.py 8 \
  --work-dir $WORK_DIR
```

There are also some args for you to fill in the config file as below.

```python
"""
_base_(str): The path to your pruning config file.
pruned_path (str): The path to the checkpoint of the pruned model.
finetune_lr (float): The lr rate to finetune. Usually, we directly use the lr
    rate of the pretrain.
"""
```

After finetuning, except a checkpoint of the best model, there is also a fix_subnet.json, which records the pruned model structure. It will be used when deploying.

### Test

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 ./tools/dist_test.sh \
   {config_folder}/group_fisher_{normalization_type}_finetune_{model_name}.py {checkpoint_path} 8
```

### Deploy

For a pruned model, you only need to use the pruning deploy config to instead the pretrain config to deploy the pruned version of your model. If you are not familiar with mmdeploy, it's recommended to refer to [MMDeploy document](https://mmdeploy.readthedocs.io/en/latest/02-how-to-run/convert_model.html).

```bash
python {mmdeploy}/tools/deploy.py \
    {mmdeploy}/{mmdeploy_config}.py \
    {config_folder}/group_fisher_{normalization_type}_deploy_{model_name}.py \
    {path_to_finetuned_checkpoint}.pth \
    {mmdeploy}/tests/data/tiger.jpeg
```

The deploy config has some args as below:

```python
"""
_base_ (str): The path to your pretrain config file.
fix_subnet (Union[dict,str]): The dict store the pruning structure or the
    json file including it.
divisor (int): The divisor the make the channel number divisible.
"""
```

The divisor is important for the actual inference speed, and we suggest you to test it in \[1,2,4,8,16,32\] to find the fastest divisor.

## Reference

[GroupFisher in MMRazor](https://github.com/open-mmlab/mmrazor/tree/main/configs/pruning/base/group_fisher)

[rp_sa_f]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_finetune_rtmpose-s_8xb256-420e_aic-coco-256x192.pth
[rp_sa_l]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_finetune_rtmpose-s_8xb256-420e_aic-coco-256x192.json
[rp_sa_p]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_prune_rtmpose-s_8xb256-420e_aic-coco-256x192.pth
[rp_sc_f]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_finetune_rtmpose-s_8xb256-420e_coco-256x192.pth
[rp_sc_l]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_finetune_rtmpose-s_8xb256-420e_coco-256x192.json
[rp_sc_p]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_prune_rtmpose-s_8xb256-420e_coco-256x192.pth
