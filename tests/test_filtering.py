from unittest import TestCase
from pytools_image_processing.filtering import (
    threshold_percentage,
    threshold_absolute,
    adaptive_threshold,
    hsv_color_filter,
)
from pytools_image_processing.exceptions import (
    ImageNotBinaryError,
    ImageNotGrayscaleError,
    ImageNotRGBError,
)
import numpy as np
import cv2


class TestThresholdPercentage(TestCase):
    """Test the threshold function."""

    def setUp(self):
        test_file = "tests/test_images/test_analysis_lines.jpg"
        self.image = cv2.imread(test_file)
        self.gray_image = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)

    def test_runs_without_errors(self):
        """Test if the function runs without errors."""
        threshold_percentage(image=self.gray_image, thresh_value=0.5, show_steps=False)

        with self.assertRaises(ImageNotGrayscaleError):
            threshold_percentage(image=self.image, thresh_value=0.5, show_steps=False)


class TestThresholdAbsolute(TestCase):
    """Test the threshold function."""

    def setUp(self):
        test_file = "tests/test_images/test_analysis_lines.jpg"
        self.image = cv2.imread(test_file)
        self.gray_image = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)

    def test_runs_without_errors(self):
        """Test if the function runs without errors."""
        threshold_absolute(image=self.gray_image, thresh_value=127, show_steps=False)

        with self.assertRaises(ImageNotGrayscaleError):
            threshold_absolute(image=self.image, thresh_value=127, show_steps=False)


class TestAdaptiveThreshold(TestCase):
    """Test the threshold function."""

    def setUp(self):
        test_file = "tests/test_images/test_analysis_lines.jpg"
        self.image = cv2.imread(test_file)
        self.gray_image = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)

    def test_runs_without_errors(self):
        """Test if the function runs without errors."""
        adaptive_threshold(image=self.gray_image, correction=5, show_steps=False)

        with self.assertRaises(ImageNotGrayscaleError):
            adaptive_threshold(image=self.image, correction=5, show_steps=False)


class TestHSVColorFilter(TestCase):
    """Test the threshold function."""

    def setUp(self):
        test_file = "tests/test_images/test_analysis_lines.jpg"
        self.image = cv2.imread(test_file)
        self.gray_image = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def test_runs_without_errors(self):
        """Test if the function runs without errors."""
        hsv_color_filter(
            image=self.hsv_image,
            show_steps=False,
            lower_bound=(20, 20, 20),
            upper_bound=(127, 127, 127),
        )
