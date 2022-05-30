# Multi-modal Training and Uni-modal Testing (MTUT) for gesture recognition

MUTU method uses multi-modal data in training scheme, such as rgb videos and depth videos.
For each modality, an I3D network is trained to conduct gesture recognition. The property
of spatial-temporal semantic alignment across multi-modal data is utilized to supervise the
learning, in order to improve the performance of each I3D network for single modality.

In the testing phase, uni-modal data, generally rgb video, is used.
