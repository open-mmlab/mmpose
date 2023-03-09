# Tutorial 7：Develop Applications with Webcam API

MMPose Webcam API is a toolkit to develop pose-empowered applications. This tutorial introduces the features and usage of Webcam API. More technical details can be found at [API Reference](https://mmpose.org.readthedocs.build/en/latest/api.html#mmpose-apis-webcam).

<!-- TOC -->

- [Overview](#overview)
- [An Example of Webcam Applications](#an-example-of-webcam-applications)
  - [Run the demo](#run-the-demo)
  - [Configs](#configs)
    - [Buffer configurations](#buffer-configurations)
    - [Hot-key configurations](#hot-key-configurations)
  - [Architecture of a webcam application](#architecture-of-a-webcam-application)
- [Extending Webcam API with Custom Nodes](#extending-webcam-api-with-custom-nodes)
  - [Custom nodes for general functions](#custom-nodes-for-general-functions)
    - [Inherit from Node class](#inherit-from-node-class)
    - [Implement \_\_init\_\_() method](#implement-init-method)
    - [Implement process() method](#implement-process-method)
    - [Implement bypass() method](#implement-bypass-method)
  - [Custom nodes for visualization](#custom-nodes-for-visualization)
    - [Inherit from BaseVisualizerNode class](#inherit-from-basevisualizernode-class)
    - [Implement draw() method](#implement-draw-method)

<!-- TOC -->

## Overview

<div align="center">
  <img src="https://user-images.githubusercontent.com/15977946/171577402-b28e03fa-81bd-4711-9fb8-77bcd706f7c4.png">
</div>
<div align="center">
Figure 1. Overview of Webcam API
</div>

Webcam API is composed of the following main modules (Shown in Fig. 1):

1. **WebcamExecutor** (See [webcam_executor.py](/mmpose/apis/webcam/webcam_executor.py)): The interface to build and launch the application program, and perform video capturing and displaying. Besides, `WebcamExecutor` builds a certain number of functional modules according to the config to perform different basic functions like model inference, data processing, logical decision, and image drawing. when launched, the `WebcamExecutor` continually reads video frames, controls the data flow among all function modules, and finally displays the processed results. And below are concepts related to `WebcamExecutor`:
   1. **Config** : The configuration file contains the parameters of the **WebcamExecutor** and all function modules. Webcam API uses python files as configs, following the common practice of OpenMMLab;
   2. **Launcher** (e.g. [webcam_demo.py](mmpose/demo/webcam_demo.py)): A script to load the config file, build `WebcamExecutor` and invoke its `run()` method to start the application program;
2. **Node** (See [node.py](/mmpose/apis/webcam/nodes/node.py)): The interface of function module. One node usually implements a basic function. For example, `DetectorNode` performs object detection from the frame; `ObjectVisualizerNode` draws the bbox and keypoints of objects; `RecorderNode` write the frames into a local video file. Users can also add custom nodes by inheriting the `Node` interface.
3. **Utils**: Utility modules and functions including:
   1. **Message** (See [message.py](/mmpose/apis/webcam/utils/message.py)): The data interface of the `WebcamExecutor` and `Node`. `Message` instances may contain images, model inference results, text information, or arbitrary custom data;
   2. **Buffer** (See [buffer.py](/mmpose/apis/webcam/utils/buffer.py)): The container of `Message` instances for asynchronous communication between nodes. A node fetches the input from its input buffers once it's ready, and put the output into its output buffer;
   3. **Event** (See [event.py](/mmpose/apis/webcam/utils/event.py)): The event manager supports event communication within the program. Different from the data message that follows a route defined by the config, an event can be set or responded by the executor or nodes immediately. For example, when the user presses a key on the keyboard, an event will be broadcasted to all nodes. This mechanism is useful in user interaction functions.

## An Example of Webcam Applications

In this section, we will introduce how to build an application by Webcam API via a simple example.

### Run the demo

Before we dive into technical details, you can try running this demo first with the following command. What it does is read the video stream from the webcam, display it on the screen and save it to a local file.

```shell
# python demo/webcam_demo.py --config CONFIG_PATH [--debug]
python demo/webcam_demo.py --config demo/webcam_cfg/test_camera.py
```

### Configs

Now let's look at the config used in this demo:

```python
executor_cfg = dict(
    name='Test Webcam', # name of the application
    camera_id=0,  # camera ID (optionally, it can be a path of an input video file)
    camera_max_fps=30,  # maximum FPS to read the video
    nodes=[
        # `MonitorNode` shows the system and application information
        dict(
            type='MonitorNode',  # node type
            name='monitor',  # node name
            enable_key='m',  # hot key to switch on/off
            enable=False,  # init status of on/off
            input_buffer='_frame_',  # input buffer
            output_buffer='display'),  # output buffer
        # `RecorderNode` saves output to a local file
        dict(
            type='RecorderNode',  # node name
            name='recorder',  # node type
            out_video_file='webcam_output.mp4',  # path to save output
            input_buffer='display',  # input buffer
            output_buffer='_display_') # output buffer
    ])
```

As shown above, the content of the config file is a dict named `executor_cfg`, which contains basic parameters (e.g. `name`, `camera_id`, et al. See the [document](https://mmpose.readthedocs.build/en/latest/generated/mmpose.apis.webcam.WebcamExecutor.html#mmpose.apis.webcam.WebcamExecutor) for details) and node configs (`nodes`). The node configs are stored in a list, of which each element is a dict that contains parameters of one node. There are 2 nodes in the demo, namely a `DetectorNode` and a `RecorderNode`. See the [document of node](https://mmpose.readthedocs.build/en/latest/api.html#nodes) for more information.

#### Buffer configurations

From the demo config, you may have noticed that nodes usually have a special type of parameters: input and output buffers. As noted previously, a buffer is a data container to hold the input and output of nodes. And in the config, we can specify the input and output buffer of each node by buffer names. In the demo config, for example, `MonitorNode` fetches input from a buffer named `"_frame"_`, and puts output to a buffer named `"display"`; and `RecorderNode` fetches input from the buffer `"display"`, and outputs to another buffer `"_display_"`.

In the config, you can assign arbitrary buffer names, and the executor will build buffers accordingly and connect them with the nodes. It's important to note that the following 3 names are reserved for special buffers to exchange data between the executor and nodes:

- `"_input_"`: The buffer to store frames read by the executor for model inference;
- `"_frame_"`: The buffer to store frames read by the executor (same as `"_input_"`) for visualization functions. We use separate inputs for model inference and visualization so they can run asynchronously.
- `"_display_"`: The buffer to store output that has been processed by nodes. The executor will load from this buffer to display.

In an application, the executor will build a **BufferManager** instance to hold all buffers (See [`BufferManager` document](https://mmpose.readthedocs.build/en/latest/generated/mmpose.apis.webcam.utils.BufferManager.html#mmpose.apis.webcam.utils.BufferManager) for details).

#### Hot-key configurations

Some nodes support switch state control by hot-keys. These nodes have the following parameters:

- `enable_key` (str): Specify the hot-key for switch state control;
- `enable` (bool): Set the initial switch state.

The hot-key response is supported by the event mechanism. The executor has a **EvenetManager** (See [`EventManager` document](https://mmpose.readthedocs.build/en/latest/generated/mmpose.apis.webcam.utils.EventManager.html#mmpose.apis.webcam.utils.EventManager)) instance to manage all user-defined events in the application. A node can register events at initialization. Registered events can be set, waited, or cleared at run time.

### Architecture of a webcam application

Now we have introduced the concept of WebcamExecutor, Node, Buffer, and Event. The architecture of a webcam application can be illustrated as shown in Fig. 2.

<div align="center">
  <img src="https://user-images.githubusercontent.com/15977946/171552368-f961dc13-cc70-4960-bbfd-5ec791cf3b9b.png">
</div>
<div align="center">
Figure 2. Architecture of a webcam application
</div>

## Extending Webcam API with Custom Nodes

Webcam API provides a simple and efficient interface to extend by defining new nodes. In this section, we will show you how to do this via examples.

### Custom nodes for general functions

We first introduce the general steps to define new nodes. Here we take `DetectorNode` as an example.

#### Inherit from `Node` class

All node classes should inherit from the base class `Node` (See [node.py](/mmpose/apis/webcam/nodes/node.py)) and be registered to the registry `NODES`. So the node instances can be built from configs.

```python
from mmpose.apis.webcam.nodes import Node, NODES

@NODES.register_module()
class DetectorNode(Node):
    ...
```

#### Implement `__init__()` method

The `__init__()` method of `DetectorNode` is impolemented as below:

```python
    def __init__(self,
                 name: str,
                 model_config: str,
                 model_checkpoint: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 device: str = 'cuda:0',
                 bbox_thr: float = 0.5):

        # Initialize the base class
        super().__init__(name=name, enable_key=enable_key, enable=enable)

        # Initialize parameters
        self.model_config = get_config_path(model_config, 'mmdet')
        self.model_checkpoint = model_checkpoint
        self.device = device.lower()
        self.bbox_thr = bbox_thr

        self.model = init_detector(
            self.model_config, self.model_checkpoint, device=self.device)

        # Register input/output buffers
        self.register_input_buffer(input_buffer, 'input', trigger=True)  # Set trigger
        self.register_output_buffer(output_buffer)
```

The `__init__()` method usually does the following steps:

1. **Initialize the base class**: Call `super().__init__()` with parameters like `name`, `enable_key` and `enable`;
2. **Initialize node parameters**: In this example, we initializes the parameters like `model_config`, `device`, `bbox_thr` in the node, and load the model with MMDetection APIs.
3. **Register buffers**: A node needs to register its input and output buffers during initialization:
   1. Register each input buffer by `register_input_buffer()` method. This method maps the buffer name (i.e. `input_buffer` from the config) to an indicator (i.e. `"input"` in the example). At runtime, the node can access the data from the registered buffers by indicators (See [Implement process() method](#implement-process-method)).
   2. Register the output buffers by `register_output_buffer()` method. At runtime, the node output will be stored in every registered output buffer (each buffer will store a deep copy of the node output).

#### Implement `process()` method

The `process()` method defines the behavior of a node. In the `DetectorNode` example, we implement detection model inference in the `process()` method:

```python
    def process(self, input_msgs):

        # Get the input message from the buffer by the indicator 'input'
        input_msg = input_msgs['input']

        # Get image data from the input message
        img = input_msg.get_image()

        # Process model inference using MMDetection API
        preds = inference_detector(self.model, img)
        objects = self._post_process(preds)

        # Assign the detection results into the message
        input_msg.update_objects(objects)

        # Return the message
        return input_msg
```

The `process()` method usually does the following steps:

1. **Get input data**: The argument `input_msgs` contains data fetched from all registered input buffers. Data from a specific buffer can be obtained by the indicator (e.g. `"input"`);
2. **Parse input data**: The input data are usually `FrameMessage` instances (See the [document](https://mmpose.readthedocs.build/en/latest/generated/mmpose.apis.webcam.utils.FrameMessage.html#mmpose.apis.webcam.utils.FrameMessage) for details). The node can extract the image data and model inference results from the message;
3. **Process**: In this example, we use MMDetection APIs to detect objects from the input image, and post-process the result format;
4. **Return results**: The detection results are assigned to the `input_msg` by the `update_objects()` method. Then the message is returned by `process()` and will be stored in all registered output buffers to serve as the input of downstream nodes.

#### Implement `bypass()` method

If a node supports switch state control by hot-keys, its `bypass()` method should be implemented to define the node behavior when turned off. The `bypass()` method has the same function signature as the `process()` method. `DetectorNode` simply outputs the input message in the `bypass()` method as the following:

```python
    def bypass(self, input_msgs):
        return input_msgs['input']
```

### Custom nodes for visualization

**Visualizer Node** is a special category of nodes for visualization functions. Here we will introduce a simpler interface to extend this kind of nodes. We take `NoticeBoardNode` as an example, whose function is to show text information in the output frames.

#### Inherit from `BaseVisualizerNode` class

`BaseVisualizerNode` is a subclass of `Node` that partially implements the `process()` method and exposes the `draw()` method as an image editing interface. Visualizer nodes should inherit from `BaseVisualizerNode` and be registered to the registry `NODES`.

```python
from mmpose.apis.webcam.nodes import BaseVisualizerNode, NODES

@NODES.register_module()
class NoticeBoardNode(BaseVisualizerNode):
    ...
```

The implementation of `__init__()` in visualizer nodes is similar to it in general nodes. Please refer to [Implement \_\_init\_\_() method](#implement-init-method). Note that a visualizer node should register one and only one input buffer with the name `"input"`.

#### Implement `draw()` method

The `draw()` method has one argument `input_msg`, which is the data fetched from the buffer indicated by `"input"`. The return value of `draw()` is an image in `np.ndarray` type, which will be used to update the image data in `input_msg`. And the updated `input_msg` will be the node output.

We implement the `draw()` method of `NoticeBoardNode` as the following:

```python
    def draw(self, input_msg: FrameMessage) -> np.ndarray:
        # Get frame image data
        img = input_msg.get_image()

        # Create a canvas
        canvas = np.full(img.shape, self.background_color, dtype=img.dtype)

        # Put the text on the canvas image
        x = self.x_offset
        y = self.y_offset
        max_len = max([len(line) for line in self.content_lines])

        def _put_line(line=''):
            nonlocal y
            cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                        self.text_scale, self.text_color, 1)
            y += self.y_delta

        for line in self.content_lines:
            _put_line(line)

        # Copy and paste the valid region of the canvas to the frame image
        x1 = max(0, self.x_offset)
        x2 = min(img.shape[1], int(x + max_len * self.text_scale * 20))
        y1 = max(0, self.y_offset - self.y_delta)
        y2 = min(img.shape[0], y)

        src1 = canvas[y1:y2, x1:x2]
        src2 = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)

        # Return the processed image
        return img
```
