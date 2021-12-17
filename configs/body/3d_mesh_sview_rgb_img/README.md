# Human Body 3D Mesh Recovery

This task aims at recovering the full 3D mesh representation (parameterized by shape and 3D joint angles) of a
human body from a single RGB image.

## Data preparation

The preparation for human mesh recovery mainly includes:

- Datasets
- Annotations
- SMPL Model

Please follow [DATA Preparation](/docs/en/tasks/3d_body_mesh.md) to prepare them.

## Prepare Pretrained Models

Please download the pretrained HMR model from
[here](https://download.openmmlab.com/mmpose/mesh/hmr/hmr_mesh_224x224-c21e8229_20201015.pth),
and make it looks like this:

```text
mmpose
`-- models
    `-- pytorch
         `-- hmr
            |-- hmr_mesh_224x224-c21e8229_20201015.pth
```

## Inference with pretrained models

### Test a Dataset

You can use the following commands to test the pretrained model on Human3.6M test set and
evaluate the joint error.

```shell
# single-gpu testing
python tools/test.py configs/mesh/hmr/hmr_resnet_50.py \
models/pytorch/hmr/hmr_mesh_224x224-c21e8229_20201015.pth --eval=joint_error

# multiple-gpu testing
./tools/dist_test.sh configs/mesh/hmr/hmr_resnet_50.py \
models/pytorch/hmr/hmr_mesh_224x224-c21e8229_20201015.pth 8 --eval=joint_error
```

## Train the model

In order to train the model, please download the
[zip file](https://drive.google.com/file/d/1JrwfHYIFdQPO7VeBEG9Kk3xsZMVJmhtv/view?usp=sharing)
of the sampled train images of Human3.6M dataset.
Extract the images and make them look like this：

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── h36m_train
        ├── S1
        │   ├── S1_Directions_1.54138969
        │   │   ├── S1_Directions_1.54138969_000001.jpg
        │   │   ├── S1_Directions_1.54138969_000006.jpg
        │   │   └── ...
        │   ├── S1_Directions_1.55011271
        │   └── ...
        ├── S11
        │   ├── S11_Directions_1.54138969
        │   ├── S11_Directions_1.55011271
        │   └── ...
        ├── S5
        │   ├── S5_Directions_1.54138969
        │   ├── S5_Directions_1.55011271
        │   └── S5_WalkTogether.60457274
        ├── S6
        │   ├── S6_Directions_1.54138969
        │   ├── S6_Directions_1.55011271
        │   └── S6_WalkTogether.60457274
        ├── S7
        │   ├── S7_Directions_1.54138969
        │   ├── S7_Directions_1.55011271
        │   └── S7_WalkTogether.60457274
        ├── S8
        │   ├── S8_Directions_1.54138969
        │   ├── S8_Directions_1.55011271
        │   └── S8_WalkTogether_2.60457274
        └── S9
            ├── S9_Directions_1.54138969
            ├── S9_Directions_1.55011271
            └── S9_WalkTogether.60457274

```

Please also download the preprocessed annotation file for Human3.6M train set from
[here](https://drive.google.com/file/d/1NveJQGS4IYaASaJbLHT_zOGqm6Lo_gh5/view?usp=sharing)
under `$MMPOSE/data/mesh_annotation_files`, and make it like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── mesh_annotation_files
        ├── h36m_train.npz
        └── ...
```

### Train with multiple GPUs

Here is the code of using 8 GPUs to train HMR net:

```shell
./tools/dist_train.sh configs/mesh/hmr/hmr_resnet_50.py 8 --work-dir work_dirs/hmr --no-validate
```
