from unittest import TestCase
from pytools_image_processing.morphologic import (
    remap_image_intensity,
    normalize_image,
    change_brightness,
    change_saturation,
    average_blur,
    gaussian_blur,
)
import numpy as np
import cv2


class TestRemapImageIntensity(TestCase):
    """Test the remap_image_intensity function."""

    def test_runs_without_errors(self):
        """Test if the function runs without errors."""
        test_image = np.random.randint(0, 255, (100, 100)).astype(np.uint8)
        remap_image_intensity(image=test_image, range=(0, 255), show_steps=False)


class TestNormalizeImage(TestCase):
    """Test the normalize_image function."""

    def test_runs_without_errors(self):
        """Test if the function runs without errors."""
        test_image = np.random.randint(0, 255, (100, 100)).astype(np.uint8)
        normalize_image(image=test_image)


class TestChangeBrightness(TestCase):
    """Test the change_brightness function."""

    def test_runs_without_errors(self):
        """Test if the function runs without errors."""
        test_rgb_image = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
        change_brightness(image=test_rgb_image, delta=10, show_steps=False)


class TestChangeSaturation(TestCase):
    """Test the change_saturation function."""

    def test_runs_without_errors(self):
        """Test if the function runs without errors."""
        test_image = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
        change_saturation(image=test_image, delta=10, show_steps=False)


class TestAverageBlur(TestCase):
    """Test the blur function."""

    def test_runs_without_errors(self):
        """Test if the function runs without errors."""
        test_image = np.random.randint(0, 255, (100, 100)).astype(np.uint8)
        average_blur(image=test_image, kernel_size=3, show_steps=False)


class TestGaussianBlur(TestCase):

    def test_runs_without_errors(self):
        """Test if the function runs without errors."""
        test_image = np.random.randint(0, 255, (100, 100)).astype(np.uint8)
        gaussian_blur(image=test_image, kernel_size=3, show_steps=False)
