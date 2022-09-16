# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import cv2
import numpy as np

from mmpose.apis.webcam.utils.image_capture import ImageCapture


class TestImageCapture(unittest.TestCase):

    def setUp(self):
        self.image_path = 'tests/data/coco/000000000785.jpg'
        self.image = cv2.imread(self.image_path)

    def test_init(self):
        image_cap = ImageCapture(self.image_path)
        self.assertIsInstance(image_cap.image, np.ndarray)

        image_cap = ImageCapture(self.image)
        self.assertTrue((self.image == image_cap.image).all())

    def test_image_capture(self):
        image_cap = ImageCapture(self.image_path)

        # test operations
        self.assertTrue(image_cap.isOpened())

        flag, image_ = image_cap.read()
        self.assertTrue(flag)
        self.assertTrue((self.image == image_).all())

        image_cap.release()
        self.assertIsInstance(image_cap.image, np.ndarray)

        img_h = image_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.assertAlmostEqual(img_h, self.image.shape[0])
        img_w = image_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.assertAlmostEqual(img_w, self.image.shape[1])
        fps = image_cap.get(cv2.CAP_PROP_FPS)
        self.assertTrue(np.isnan(fps))

        with self.assertRaises(NotImplementedError):
            _ = image_cap.get(-1)


if __name__ == '__main__':
    unittest.main()
