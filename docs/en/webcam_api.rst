mmpose.apis.webcam
--------------------
.. contents:: MMPose Webcam API: Tools to build simple interactive webcam applications and demos
    :depth: 2
    :local:
    :backlinks: top

Executor
^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: mmpose.apis.webcam
.. autosummary::
    :toctree: generated
    :nosignatures:

    WebcamExecutor

Nodes
^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: mmpose.apis.webcam.nodes

Base Nodes
""""""""""""""""""""
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: webcam_node_class.rst

    Node
    BaseVisualizerNode

Model Nodes
""""""""""""""""""""
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: webcam_node_class.rst

    DetectorNode
    TopdownPoseEstimatorNode

Visualizer Nodes
""""""""""""""""""""
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: webcam_node_class.rst

    ObjectVisualizerNode
    NoticeBoardNode
    SunglassesEffectNode
    BigeyeEffectNode

Helper Nodes
""""""""""""""""""""
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: webcam_node_class.rst

    ObjectAssignerNode
    MonitorNode
    RecorderNode

Utils
^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: mmpose.apis.webcam.utils

Buffer and Message
""""""""""""""""""""
.. autosummary::
    :toctree: generated
    :nosignatures:

    BufferManager
    Message
    FrameMessage
    VideoEndingMessage

Pose
""""""""""""""""""""
.. autosummary::
    :toctree: generated
    :nosignatures:

    get_eye_keypoint_ids
    get_face_keypoint_ids
    get_hand_keypoint_ids
    get_mouth_keypoint_ids
    get_wrist_keypoint_ids

Event
""""""""""""""""""""
.. autosummary::
    :toctree: generated
    :nosignatures:

    EventManager

Misc
""""""""""""""""""""
.. autosummary::
    :toctree: generated
    :nosignatures:

    copy_and_paste
    screen_matting
    expand_and_clamp
    limit_max_fps
    is_image_file
    get_cached_file_path
    load_image_from_disk_or_url
    get_config_path
