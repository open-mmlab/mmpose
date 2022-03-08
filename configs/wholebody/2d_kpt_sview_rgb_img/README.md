# 2D Human Whole-Body Pose Estimation

2D human whole-body pose estimation aims to localize dense landmarks on the entire human body including face, hands, body, and feet.

Existing approaches can be categorized into top-down and bottom-up approaches.

Top-down methods divide the task into two stages: human detection and whole-body pose estimation. They perform human detection first, followed by single-person whole-body pose estimation given human bounding boxes.

Bottom-up approaches (e.g. AE) first detect all the whole-body keypoints and then group/associate them into person instances.

## Data preparation

Please follow [DATA Preparation](/docs/en/tasks/2d_wholebody_keypoint.md) to prepare data.

## Demo

Please follow [Demo](/demo/docs/2d_wholebody_pose_demo.md) to run demos.

<img src="https://user-images.githubusercontent.com/9464825/95552839-00a61080-0a40-11eb-818c-b8dad7307217.gif" width="600px" alt><br>
