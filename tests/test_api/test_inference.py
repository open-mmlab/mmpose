from mmpose.apis import (inference_bottom_up_pose_model,
                         inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)


def test_top_down_demo():
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/top_down/resnet/coco/res50_coco_256x192.py', None)

    image_name = 'tests/data/coco/000000000785.jpg'
    # test a single image, with a list of bboxes.
    pose_results = inference_top_down_pose_model(
        pose_model, image_name, [[50, 50, 50, 100]], format='xywh')

    # show the results
    vis_pose_result(pose_model, image_name, pose_results, skeleton=skeleton)


def test_bottom_up_demo():
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/bottom_up/resnet/coco/res50_coco_512x512.py', None)

    image_name = 'tests/data/coco/000000000785.jpg'

    pose_results = inference_bottom_up_pose_model(pose_model, image_name)

    # show the results
    vis_pose_result(pose_model, image_name, pose_results, skeleton=skeleton)
