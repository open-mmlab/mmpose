# Human Mesh Estimation

## Data preparation
### Human3.6M

For Human3.6M, we use the MoShed data provided in [HMR](https://github.com/akanazawa/hmr) for training.
However, due to license limitations, we are not allowed to redistribute the MoShed data.

For the evaluation on Human3.6M dataset, please follow the
[preprocess procedure](https://github.com/nkolot/SPIN/tree/master/datasets/preprocess)
of SPIN to extract test images from
[Human3.6M](http://vision.imar.ro/human3.6m/description.php) original videos,
and make it look like this:

```
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── Human3.6M
        ├── images
            ├── S11_Directions_1.54138969_000001.jpg
            ├── S11_Directions_1.54138969_000006.jpg
            ├── S11_Directions_1.54138969_000011.jpg
            ├── ...
```

The download of Human3.6M dataset is quite difficult, you can also download the
[zip file](https://drive.google.com/file/d/1WnRJD9FS3NUf7MllwgLRJJC-JgYFr8oi/view?usp=sharing)
of the test images. However, due to the license limitations, we are not allowed to
redistribute the images either. So the users need to download the original video and
extract the images by themselves.

### MPI-INF-3DHP

For MPI-INF-3DHP, please follow the
[preprocess procedure](https://github.com/nkolot/SPIN/tree/master/datasets/preprocess)
of SPIN to sample images, and make them like this:

```
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    ├── mpi_inf_3dhp_test_set
    │   ├── TS1
    │   ├── TS2
    │   ├── TS3
    │   ├── TS4
    │   ├── TS5
    │   └── TS6
    ├── S1
    │   ├── Seq1
    │   └── Seq2
    ├── S2
    │   ├── Seq1
    │   └── Seq2
    ├── S3
    │   ├── Seq1
    │   └── Seq2
    ├── S4
    │   ├── Seq1
    │   └── Seq2
    ├── S5
    │   ├── Seq1
    │   └── Seq2
    ├── S6
    │   ├── Seq1
    │   └── Seq2
    ├── S7
    │   ├── Seq1
    │   └── Seq2
    └── S8
        ├── Seq1
        └── Seq2
```

### LSP

For LSP, please download the high resolution version
[LSP dataset original](http://sam.johnson.io/research/lsp_dataset_original.zip).
Extract them under `$MMPOSE/data`, and make them look like this:

```
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── lsp_dataset_original
        ├── images
            ├── im0001.jpg
            ├── im0002.jpg
            └── ...
```

### LSPET

For LSPET, please download its high resolution form
[HR-LSPET](http://datasets.d2.mpi-inf.mpg.de/hr-lspet/hr-lspet.zip).
Extract them under `$MMPOSE/data`, and make them look like this:

```
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── lspet_dataset
        ├── images
        │   ├── im00001.jpg
        │   ├── im00002.jpg
        │   ├── im00003.jpg
        │   └── ...
        └── joints.mat
```

### Annotation Files for Human Mesh Estimation

For human mesh estimation, we use multiple datasets for training.
The annotation of different datasets are preprocessed to the same format. Please
follow the [preprocess procedure](https://github.com/nkolot/SPIN/tree/master/datasets/preprocess)
of SPIN to generate the annotation files or download the processed files from
[here](https://download.openmmlab.com/mmpose/datasets/mesh_annotation_files.zip),
and make it look like this:

```
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── mesh_annotation_files
        ├── coco_2014_train.npz
        ├── h36m_valid_protocol1.npz
        ├── h36m_valid_protocol2.npz
        ├── hr-lspet_train.npz
        ├── lsp_dataset_original_train.npz
        ├── mpi_inf_3dhp_train.npz
        └── mpii_train.npz
```

### CMU MoShed Data

Real-world SMPL parameters are used for the adversarial training in human mesh estimation.
The MoShed data provided in [HMR](https://github.com/akanazawa/hmr) is included in this
[zip file](https://download.openmmlab.com/mmpose/datasets/mesh_annotation_files.zip).
Please download and extract it under `$MMPOSE/data`, and make it look like this:

```
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── mesh_annotation_files
        ├── CMU_mosh.npz
        └── ...
```

### SMPL Model

For human mesh estimation, SMPL model is used to generate the human mesh.
Please download the [gender neutral SMPL model](http://smplify.is.tue.mpg.de/),
[joints regressor](https://download.openmmlab.com/mmpose/datasets/joints_regressor_cmr.npy)
and [mean parameters](https://download.openmmlab.com/mmpose/datasets/smpl_mean_params.npz)
under `$MMPOSE/models/smpl`, and make it look like this:

```
mmpose
├── mmpose
├── ...
├── models
    │── smpl
        ├── joints_regressor_cmr.npy
        ├── smpl_mean_params.npz
        └── SMPL_NEUTRAL.pkl
```

## Prepare Pretrained Models
Download the pretrained HMR model from
[here](https://download.openmmlab.com/mmpose/mesh/hmr/hmr_mesh_224x224-c21e8229_20201015.pth).

```
mmpose
`-- models
    `-- pytorch
         `-- hmr
            |-- hmr_mesh_224x224-c21e8229_20201015.pth
            |-- ...
```

## Inference with pretrained models

### Test a dataset
You can use the following commands to test on Human3.6M test set and
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
Inorder to train the model, please download the
[zip file](https://drive.google.com/file/d/1JrwfHYIFdQPO7VeBEG9Kk3xsZMVJmhtv/view?usp=sharing)
of the sampled train images of Human3.6M dataset.
Extract the images and make them look like this：

```
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

```
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
