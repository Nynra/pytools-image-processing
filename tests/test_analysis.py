from unittest import TestCase
from pytools_image_processing.analysis import (
    find_components,
    plot_intensity_profile,
    find_edges,
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
        self.image_path = os.path.join("tests", "test_images", "test_analysis_lines.jpg")
        self.assertTrue(os.path.exists(self.image_path))
        self.image = cv2.imread(
            self.image_path,
            cv2.IMREAD_COLOR
        )
        self.gray_image = cv2.imread(
            self.image_path,
            cv2.IMREAD_GRAYSCALE
        )

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

    def test_find_edges_grayscale(self):
        """Test if the function runs without errors."""
        edge_img = find_edges(image=self.gray_image, show_steps=False)

        with self.assertRaises(ImageNotGrayscaleError):
            find_edges(image=self.image, show_steps=False)


class TestFindComponents(TestCase):

    def setUp(self):
        # The function expects a mask so we have to threshold the image first
        self.image_path = os.path.join("tests", "test_images", "test_analysis_lines.jpg")
        self.assertTrue(os.path.exists(self.image_path))
        self.gray_image = cv2.imread(
            self.image_path,
            cv2.IMREAD_GRAYSCALE
        )
        # Use inverted binary, otherwise the background is not 0
        _, self.binary_image = cv2.threshold(
            self.gray_image, 150, 255, cv2.THRESH_BINARY_INV
        )

        # Predetermined parameters for this image
        self.min_size = 5000
        self.max_size = 1000000
        self.expected_components = 4

    def test_runs_without_errors(self):
        components, total_components = find_components(
            image=self.binary_image, min_size=self.min_size, max_size=self.max_size, show_steps=False
        )

        with self.assertRaises(ImageNotBinaryError):
            find_components(
                image=self.gray_image, min_size=self.min_size, max_size=self.max_size, show_steps=False
            )

    def test_find_components(self):
        components, total_components = find_components(
            image=self.binary_image, min_size=self.min_size, max_size=self.max_size, show_steps=False
        )
        self.assertEqual(len(components), self.expected_components)
