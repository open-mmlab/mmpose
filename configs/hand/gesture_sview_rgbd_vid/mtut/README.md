# Multi-modal Training and Uni-modal Testing (MTUT) for gesture recognition

MTUT method uses multi-modal data in the training phase, such as RGB videos and depth videos.
For each modality, an I3D network is trained to conduct gesture recognition. The property
of spatial-temporal semantic alignment across multi-modal data is utilized to supervise the
learning, in order to improve the performance of each I3D network for a single modality.

In the testing phase, uni-modal data, generally RGB video, is used.
