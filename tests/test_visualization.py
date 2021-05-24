import numpy as np

from mmpose.core import imshow_keypoints, imshow_keypoints_3d


def test_imshow_keypoints():
    # 2D keypoint
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    kpts = np.array([[1, 1, 1], [10, 10, 1]], dtype=np.float32)
    pose_result = [kpts]
    skeleton = [[1, 2]]
    pose_kpt_color = [(127, 127, 127)] * len(kpts)
    pose_limb_color = [(127, 127, 127)] * len(skeleton)
    img_vis_2d = imshow_keypoints(
        img,
        pose_result,
        skeleton=skeleton,
        pose_kpt_color=pose_kpt_color,
        pose_limb_color=pose_limb_color,
        show_keypoint_weight=True)

    # 3D keypoint
    kpts_3d = np.array([[0, 0, 0, 1], [1, 1, 1, 1]], dtype=np.float32)
    pose_result_3d = [{'keypoints_3d': kpts_3d, 'title': 'test'}]
    _ = imshow_keypoints_3d(
        pose_result_3d,
        img=img_vis_2d,
        skeleton=skeleton,
        pose_kpt_color=pose_kpt_color,
        pose_limb_color=pose_limb_color,
        vis_height=400)
