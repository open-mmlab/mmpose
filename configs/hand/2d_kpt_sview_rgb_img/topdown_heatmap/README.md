# Top-down heatmap-based hand keypoint estimation

Top-down methods divide the task into two stages: hand detection and hand keypoint estimation.

They perform hand detection first, followed by hand keypoint estimation given hand bounding boxes.
Instead of estimating keypoint coordinates directly, the pose estimator will produce heatmaps which represent the
likelihood of being a keypoint.

Various neural network models have been proposed for better performance.
