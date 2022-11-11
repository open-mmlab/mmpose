mmpose.apis
-------------
.. automodule:: mmpose.apis
    :members:

mmpose.codecs
-------------
.. automodule:: mmpose.codecs
    :members:

mmpose.models
---------------
backbones
^^^^^^^^^^^
.. automodule:: mmpose.models.backbones
    :members:

necks
^^^^^^^^^^^
.. automodule:: mmpose.models.necks
    :members:

detectors
^^^^^^^^^^^
.. automodule:: mmpose.models.detectors
    :members:

heads
^^^^^^^^^^^^^^^
.. automodule:: mmpose.models.heads
    :members:

losses
^^^^^^^^^^^
.. automodule:: mmpose.models.losses
    :members:

misc
^^^^^^^^^^^
.. automodule:: mmpose.models.misc
    :members:

mmpose.datasets
-----------------
.. automodule:: mmpose.datasets
    :members:

datasets
^^^^^^^^^^^
.. automodule:: mmpose.datasets.datasets.base
    :members:
    :noindex:

.. automodule:: mmpose.datasets.datasets.body
    :members:
    :noindex:

.. automodule:: mmpose.datasets.datasets.face
    :members:
    :noindex:

.. automodule:: mmpose.datasets.datasets.hand
    :members:
    :noindex:

.. automodule:: mmpose.datasets.datasets.animal
    :members:
    :noindex:

.. automodule:: mmpose.datasets.datasets.fashion
    :members:
    :noindex:

transforms
^^^^^^^^^^^
.. automodule:: mmpose.datasets.transforms.loading
    :members:

.. automodule:: mmpose.datasets.transforms.common_transforms
    :members:

.. automodule:: mmpose.datasets.transforms.topdown_transforms
    :members:

.. automodule:: mmpose.datasets.transforms.bottomup_transforms
    :members:

.. automodule:: mmpose.datasets.transforms.formatting
    :members:

mmpose.structures
---------------
.. automodule:: mmpose.structures
    :members:

bbox
^^^^^^^^^^^
.. automodule:: mmpose.structures.bbox
    :members:

keypoint
^^^^^^^^^^^
.. automodule:: mmpose.structures.keypoint
    :members:


mmpose.registry
---------------
.. automodule:: mmpose.registry
    :members:

mmpose.evaluation
-----------------
metrics
^^^^^^^^^^^
.. automodule:: mmpose.evaluation.metrics
    :members:

functional
^^^^^^^^^^^
.. automodule:: mmpose.evaluation.functional
    :members:

mmpose.visualization
--------------------
.. automodule:: mmpose.visualization
    :members:

mmpose.engine
---------------
hooks
^^^^^^^^^^^
.. automodule:: mmpose.engine.hooks
    :members:

webcam apis
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
