# Top-down heatmap-based fashion keypoint estimation

Top-down methods divide the task into two stages: clothes detection and fashion keypoint estimation.

They perform clothes detection first, followed by fashion keypoint estimation given fashion bounding boxes.
Instead of estimating keypoint coordinates directly, the pose estimator will produce heatmaps which represent the
likelihood of being a keypoint.

Various neural network models have been proposed for better performance.
