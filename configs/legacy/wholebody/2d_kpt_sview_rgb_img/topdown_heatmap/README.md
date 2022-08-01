# Top-down heatmap-based whole-body pose estimation

Top-down methods divide the task into two stages: human detection and whole-body pose estimation.

They perform human detection first, followed by single-person whole-body pose estimation given human bounding boxes.
Instead of estimating keypoint coordinates directly, the pose estimator will produce heatmaps which represent the
likelihood of being a keypoint.

Various neural network models have been proposed for better performance.
The popular ones include stacked hourglass networks, and HRNet.
