# Image-based Human Body 2D Pose Estimation

Multi-person human pose estimation is defined as the task of detecting the poses (or keypoints) of all people from an input image.

Existing approaches can be categorized into top-down and bottom-up approaches.

Top-down methods (e.g. deeppose) divide the task into two stages: human detection and pose estimation. They perform human detection first, followed by single-person pose estimation given human bounding boxes.

Bottom-up approaches (e.g. AE) first detect all the keypoints and then group/associate them into person instances.

## Data preparation

Please follow [DATA Preparation](/docs/en/tasks/2d_body_keypoint.md) to prepare data.

## Demo

Please follow [Demo](/demo/docs/2d_human_pose_demo.md#2d-human-pose-demo) to run demos.

<img src="demo/resources/demo_coco.gif" width="600px" alt>
