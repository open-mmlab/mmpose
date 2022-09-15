# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import cv2
import numpy as np

from mmpose.apis.webcam.utils.image_capture import ImageCapture


class TestImageCapture(unittest.TestCase):

    def test_image_capture(self):

        # test initialization
        image_path = 'tests/data/coco/000000000785.jpg'
        image_cap = ImageCapture(image_path)
        self.assertIsInstance(image_cap.image, np.ndarray)

        image = cv2.imread(image_path)
        image_cap = ImageCapture(image)
        self.assertTrue((image == image_cap.image).all())

        # test operations
        self.assertTrue(image_cap.isOpened())

        flag, image_ = image_cap.read()
        self.assertTrue(flag)
        self.assertTrue((image == image_).all())

        image_cap.release()
        self.assertIsInstance(image_cap.image, np.ndarray)

        img_h = image_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.assertAlmostEqual(img_h, image.shape[0])
        img_w = image_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.assertAlmostEqual(img_w, image.shape[1])
        fps = image_cap.get(cv2.CAP_PROP_FPS)
        self.assertTrue(np.isnan(fps))

        with self.assertRaises(NotImplementedError):
            _ = image_cap.get(-1)


if __name__ == '__main__':
    unittest.main()
