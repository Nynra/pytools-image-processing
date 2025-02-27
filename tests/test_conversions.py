from unittest import TestCase
from pytools_image_processing.conversions import (
    grayscale_to_fft_image,
)
from pytools_image_processing.exceptions import ImageNotGrayscaleError
import cv2
import numpy as np


class TestGrayscaleToFFTImage(TestCase):
    """Test the grayscale_to_fft_image function."""

    def setUp(self):
        self.image = cv2.imread("tests/test_analysis_lines.jpg", cv2.IMREAD_GRAYSCALE)

    def test_runs_without_errors(self):
        """Test if the function runs without errors."""
        grayscale_to_fft_image(image=self.image, show_steps=False)