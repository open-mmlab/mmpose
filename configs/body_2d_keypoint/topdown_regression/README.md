# Top-down regression-based pose estimation

Top-down methods divide the task into two stages: object detection and pose estimation.

They perform object detection first, followed by single-object pose estimation given object bounding boxes. With features extracted from the bounding box area, the model learns to directly regress the keypoint coordinates.
