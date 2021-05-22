from mmpose.datasets import DatasetInfo


def test_dataset_info():
    dataset_info = DatasetInfo('dataset_info/atrw.py')
    assert dataset_info.keypoint_num == len(dataset_info.flip_index)

    dataset_info = DatasetInfo('dataset_info/300w.py')
    assert dataset_info.keypoint_num == len(dataset_info.flip_index)
