# Copyright (c) OpenMMLab. All rights reserved.
import time
import uuid
import warnings
from typing import Dict, List, Optional

import numpy as np


class Message():
    """Message base class.

    All message class should inherit this class. The basic use of a Message
    instance is to carray a piece of text message (self.msg) and a dict that
    stores structured data (self.data), e.g. frame image, model prediction,
    et al.

    A message may also hold route information, which is composed of
    information of all nodes the message has passed through.

    Parameters:
        msg (str): The text message.
        data (dict, optional): The structured data.
    """

    def __init__(self, msg: str = '', data: Optional[Dict] = None):
        self.msg = msg
        self.data = data if data else {}
        self.route_info = []
        self.timestamp = time.time()
        self.id = uuid.uuid4()

    def update_route_info(self,
                          node=None,
                          node_name: Optional[str] = None,
                          node_type: Optional[str] = None,
                          info: Optional[Dict] = None):
        """Append new node information to the route information.

        Args:
            node (Node, optional): An instance of Node that provides basic
                information like the node name and type. Default: None.
            node_name (str, optional): The node name. If node is given,
                node_name will be ignored. Default: None.
            node_type (str, optional): The class name of the node. If node
                is given, node_type will be ignored. Default: None.
            info (dict, optional): The node information, which is usually
                given by node.get_node_info(). Default: None.
        """
        if node is not None:
            if node_name is not None or node_type is not None:
                warnings.warn(
                    '`node_name` and `node_type` will be overridden if node'
                    'is provided.')
            node_name = node.name
            node_type = node.__class__.__name__

        node_info = {'node': node_name, 'node_type': node_type, 'info': info}
        self.route_info.append(node_info)

    def set_route_info(self, route_info: List):
        """Directly set the entire route information.

        Args:
            route_info (list): route information to set to the message.
        """
        self.route_info = route_info

    def merge_route_info(self, route_info: List):
        """Merge the given route information into the original one of the
        message. This is used for combining route information from multiple
        messages. The node information in the route will be reordered according
        to their timestamps.

        Args:
            route_info (list): route information to merge.
        """
        self.route_info += route_info
        self.route_info.sort(key=lambda x: x.get('timestamp', np.inf))

    def get_route_info(self) -> List:
        return self.route_info.copy()


class VideoEndingMessage(Message):
    """A special message to indicate the input video is ending."""


class FrameMessage(Message):
    """The message to store information of a video frame.

    A FrameMessage instance usually holds following data in self.data:
        - image (array): The frame image
        - detection_results (list): A list to hold detection results of
            multiple detectors. Each element is a tuple (tag, result)
        - pose_results (list): A list to hold pose estimation results of
            multiple pose estimator. Each element is a tuple (tag, result)
    """

    def __init__(self, img):
        super().__init__(data=dict(image=img))

    def get_image(self):
        """Get the frame image.

        Returns:
            array: The frame image.
        """
        return self.data.get('image', None)

    def set_image(self, img):
        """Set the frame image to the message."""
        self.data['image'] = img

    def add_detection_result(self, result, tag: str = None):
        """Add the detection result from one model into the message's
        detection_results.

        Args:
            tag (str, optional): Give a tag to the result, which can be used
                to retrieve specific results.
        """
        if 'detection_results' not in self.data:
            self.data['detection_results'] = []
        self.data['detection_results'].append((tag, result))

    def get_detection_results(self, tag: str = None):
        """Get detection results of the message.

        Args:
            tag (str, optional): If given, only the results with the tag
                will be retrieved. Otherwise all results will be retrieved.
                Default: None.

        Returns:
            list[dict]: The retrieved detection results
        """
        if 'detection_results' not in self.data:
            return None
        if tag is None:
            results = [res for _, res in self.data['detection_results']]
        else:
            results = [
                res for _tag, res in self.data['detection_results']
                if _tag == tag
            ]
        return results

    def add_pose_result(self, result, tag=None):
        """Add the pose estimation result from one model into the message's
        pose_results.

        Args:
            tag (str, optional): Give a tag to the result, which can be used
                to retrieve specific results.
        """
        if 'pose_results' not in self.data:
            self.data['pose_results'] = []
        self.data['pose_results'].append((tag, result))

    def get_pose_results(self, tag=None):
        """Get pose estimation results of the message.

        Args:
            tag (str, optional): If given, only the results with the tag
                will be retrieved. Otherwise all results will be retrieved.
                Default: None.

        Returns:
            list[dict]: The retrieved pose results
        """
        if 'pose_results' not in self.data:
            return None
        if tag is None:
            results = [res for _, res in self.data['pose_results']]
        else:
            results = [
                res for _tag, res in self.data['pose_results'] if _tag == tag
            ]
        return results

    def get_full_results(self):
        """Get all model predictions of the message.

        See set_full_results() for inference.

        Returns:
            dict: All model predictions, including:
                - detection_results
                - pose_results
        """
        result_keys = ['detection_results', 'pose_results']
        results = {k: self.data[k] for k in result_keys}
        return results

    def set_full_results(self, results):
        """Set full model results directly.

        Args:
            results (dict): All model predictions including:
                - detection_results (list): see also add_detection_results()
                - pose_results (list): see also add_pose_results()
        """
        self.data.update(results)
