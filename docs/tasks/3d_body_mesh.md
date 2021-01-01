# 3D Body Mesh Recovery Datasets

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

To achieve high-quality human mesh estimation, we use multiple datasets for training.
The following items should be prepared for human mesh training:

<!-- TOC -->

- [3D Body Mesh Recovery Datasets](#3d-body-mesh-recovery-datasets)
  - [Datasets](#datasets)
    - [COCO](#coco)
    - [Human3.6M](#human36m)
    - [MPI-INF-3DHP](#mpi-inf-3dhp)
    - [LSP](#lsp)
    - [LSPET](#lspet)
    - [CMU MoShed Data](#cmu-moshed-data)
  - [Annotation Files for Human Mesh Estimation](#annotation-files-for-human-mesh-estimation)
  - [SMPL Model](#smpl-model)

<!-- TOC -->

## Datasets

### COCO

For [COCO](http://cocodataset.org/) data, please download from [COCO download](http://cocodataset.org/#download). COCO'2014 Train is needed for human mesh estimation training.
Download and extract them under $MMPOSE/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── coco
        │-- train2014
        │   ├── COCO_train2014_000000000009.jpg
        │   ├── COCO_train2014_000000000025.jpg
        │   ├── COCO_train2014_000000000030.jpg
        |   │-- ...

```

### Human3.6M

For [Human3.6M](http://vision.imar.ro/human3.6m/description.php), we use the MoShed data provided in [HMR](https://github.com/akanazawa/hmr) for training.
However, due to license limitations, we are not allowed to redistribute the MoShed data.

For the evaluation on Human3.6M dataset, please follow the
[preprocess procedure](https://github.com/nkolot/SPIN/tree/master/datasets/preprocess)
of SPIN to extract test images from
[Human3.6M](http://vision.imar.ro/human3.6m/description.php) original videos,
and make it look like this:

```text
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

For [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/), please follow the
[preprocess procedure](https://github.com/nkolot/SPIN/tree/master/datasets/preprocess)
of SPIN to sample images, and make them like this:

```text
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

For [LSP](https://sam.johnson.io/research/lsp.html), please download the high resolution version
[LSP dataset original](http://sam.johnson.io/research/lsp_dataset_original.zip).
Extract them under `$MMPOSE/data`, and make them look like this:

```text
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

For [LSPET](https://sam.johnson.io/research/lspet.html), please download its high resolution form
[HR-LSPET](http://datasets.d2.mpi-inf.mpg.de/hr-lspet/hr-lspet.zip).
Extract them under `$MMPOSE/data`, and make them look like this:

```text
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

### CMU MoShed Data

Real-world SMPL parameters are used for the adversarial training in human mesh estimation.
The MoShed data provided in [HMR](https://github.com/akanazawa/hmr) is included in this
[zip file](https://download.openmmlab.com/mmpose/datasets/mesh_annotation_files.zip).
Please download and extract it under `$MMPOSE/data`, and make it look like this:

```text
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

## Annotation Files for Human Mesh Estimation

For human mesh estimation, we use multiple datasets for training.
The annotation of different datasets are preprocessed to the same format. Please
follow the [preprocess procedure](https://github.com/nkolot/SPIN/tree/master/datasets/preprocess)
of SPIN to generate the annotation files or download the processed files from
[here](https://download.openmmlab.com/mmpose/datasets/mesh_annotation_files.zip),
and make it look like this:

```text
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

## SMPL Model

For human mesh estimation, SMPL model is used to generate the human mesh.
Please download the [gender neutral SMPL model](http://smplify.is.tue.mpg.de/),
[joints regressor](https://download.openmmlab.com/mmpose/datasets/joints_regressor_cmr.npy)
and [mean parameters](https://download.openmmlab.com/mmpose/datasets/smpl_mean_params.npz)
under `$MMPOSE/models/smpl`, and make it look like this:

```text
mmpose
├── mmpose
├── ...
├── models
    │── smpl
        ├── joints_regressor_cmr.npy
        ├── smpl_mean_params.npz
        └── SMPL_NEUTRAL.pkl
```
