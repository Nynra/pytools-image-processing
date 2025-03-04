from unittest import TestCase
from pytools_image_processing.analysis import (
    plot_intensity_profile,
)
from pytools_image_processing.exceptions import (
    ImageNotBinaryError,
    ImageNotGrayscaleError,
    ImageNot3ChannelError,
)
import numpy as np
import cv2
import os


class TestPlottingFunctions(TestCase):
    """Test that the plotting functions run without errors.

    .. attention::
        These test do not check if the output of the functions are correct
        only that the functions run without errors.
    """

    def setUp(self):
        self.image_path = os.path.join(
            "tests", "test_images", "test_analysis_lines.jpg"
        )
        self.assertTrue(os.path.exists(self.image_path))
        self.image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        self.gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

    def test_plot_intensity_profile(self):
        """Test if the function runs without errors."""
        plot_intensity_profile(image=self.gray_image, show_steps=False)

        with self.assertRaises(ImageNotGrayscaleError):
            plot_intensity_profile(image=self.image, show_steps=False)

    def test_get_rgb_histogram(self):
        """Test if the function runs without errors."""
        plot_intensity_profile(image=self.gray_image, show_steps=False)

        with self.assertRaises(ImageNotGrayscaleError):
            plot_intensity_profile(image=self.image, show_steps=False)
