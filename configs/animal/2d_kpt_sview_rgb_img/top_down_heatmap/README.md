# Top-down heatmap-based pose estimation

Top-down methods divide the task into two stages: object detection and pose estimation.

They perform object detection first, followed by single-object pose estimation given object bounding boxes.
Instead of estimating keypoint coordinates directly, the pose estimator will produce heatmaps which represent the
likelihood of being a keypoint.
