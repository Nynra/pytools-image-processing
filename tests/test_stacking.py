from unittest import TestCase
from pytools_image_processing.stacking import (
    stack_keypoint_matching,
    stack_ECC,
)
import numpy as np
import cv2


class TestStackKeypointMatching(TestCase):

    def test_runs_without_errors(self):
        test_images = [
            np.random.randint(0, 255, (100, 100)).astype(np.uint8)
            for _ in range(3)
        ]
        stack_keypoint_matching(images=test_images, show_steps=False)


class TestStackECC(TestCase):
    
    def test_runs_without_errors(self):
        self.fail("Function errors for a correct reason, real images are needed")
        test_images = [
            np.random.randint(0, 255, (100, 100)).astype(np.uint8)
            for _ in range(3)
        ]
        stack_ECC(images=test_images, show_steps=False)