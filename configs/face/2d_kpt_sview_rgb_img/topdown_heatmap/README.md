# Top-down heatmap-based face keypoint estimation

Top-down methods divide the task into two stages: face detection and face keypoint estimation.

They perform face detection first, followed by face keypoint estimation given face bounding boxes.
Instead of estimating keypoint coordinates directly, the pose estimator will produce heatmaps which represent the
likelihood of being a keypoint.

Various neural network models have been proposed for better performance.
The popular ones include HRNetv2.
