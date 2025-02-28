from unittest import TestCase
from pytools_image_processing.utils import (
    show_images,
    check_three_channel_image,
    check_grayscale_image,
    check_binary_image,
    load_image,
    get_bounding_rect,
    rotate_image,
    rotated_rect_with_max_area,
    crop_image,
)
from pytools_image_processing.exceptions import (
    ImageNotGrayscaleError,
    ImageNotBinaryError,
    ImageNot3ChannelError,
)
import numpy as np
import cv2
import os


class TestShowImages(TestCase):
    """Test the show_images function."""

    def test_runs_without_errors(self):
        """Test if the function runs without errors."""
        test_images = {
            "test_image": np.random.randint(0, 255, (100, 100)).astype(np.uint8)
        }
        show_images(images=test_images)


class TestCheckThreeChannelImage(TestCase):
    """Test the check_three_channel_image function."""

    def test_verify_3channel_image(self):
        """Test if the function verifies 3-channel images correctly."""
        self.assertTrue(
            check_three_channel_image(
                img=np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8),
                raise_exceptions=False,
            )
        )

    def test_3channel_or_binary(self):
        """Test if the function decides between 3-channel and binary images correctly."""
        self.assertFalse(
            check_three_channel_image(
                img=np.random.randint(0, 2, (100, 100)).astype(np.bool),
                raise_exceptions=False,
            )
        )
        with self.assertRaises(ImageNot3ChannelError):
            check_three_channel_image(
                img=np.random.randint(0, 2, (100, 100)).astype(np.bool),
                raise_exceptions=True,
            )

    def test_3channel_or_grayscale(self):
        """Test if the function decides between 3-channel and grayscale images correctly."""
        self.assertFalse(
            check_three_channel_image(
                img=np.random.randint(0, 255, (100, 100)).astype(np.uint8),
                raise_exceptions=False,
            )
        )
        with self.assertRaises(ImageNot3ChannelError):
            check_three_channel_image(
                img=np.random.randint(0, 255, (100, 100)).astype(np.uint8),
                raise_exceptions=True,
            )


class TestCheckGrayscaleImage(TestCase):
    """Test the check_grayscale_image function."""

    def test_verify_grayscale_image(self):
        """Test if the function verifies grayscale images correctly."""
        self.assertTrue(
            check_grayscale_image(
                img=np.random.randint(0, 255, (100, 100)).astype(np.uint8),
                raise_exceptions=False,
            )
        )

    def test_grayscale_or_binary(self):
        """Test if the function decides between grayscale and binary images correctly."""
        self.assertFalse(
            check_grayscale_image(
                img=np.random.randint(0, 2, (100, 100)).astype(np.bool),
                raise_exceptions=False,
            )
        )
        with self.assertRaises(ImageNotGrayscaleError):
            check_grayscale_image(
                img=np.random.randint(0, 2, (100, 100)).astype(np.bool),
                raise_exceptions=True,
            )

        self.assertTrue(
            check_grayscale_image(
                img=np.random.randint(0, 255, (100, 100)).astype(np.uint8),
                raise_exceptions=False,
                enforce_not_boolean=False
            )
        )

    def test_grayscale_or_3channel(self):
        """Test if the function decides between grayscale and 3-channel images correctly."""
        self.assertFalse(
            check_grayscale_image(
                img=np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8),
                raise_exceptions=False,
            )
        )
        with self.assertRaises(ImageNotGrayscaleError):
            check_grayscale_image(
                img=np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8),
                raise_exceptions=True,
            )


class TestCheckBinaryImage(TestCase):
    """Test the check_binary_image function."""

    def test_verify_binary_image(self):
        """Test if the function verifies binary images correctly."""
        self.assertTrue(
            check_binary_image(
                img=np.random.randint(0, 2, (100, 100)).astype(np.bool),
                raise_exceptions=False,
            )
        )

    def test_binary_or_grayscale(self):
        """Test if the function decides between binary and grayscale images correctly."""
        self.assertFalse(
            check_binary_image(
                img=np.random.randint(0, 255, (100, 100)).astype(np.uint8),
                raise_exceptions=False,
            )
        )
        with self.assertRaises(ImageNotBinaryError):
            check_binary_image(
                img=np.random.randint(0, 255, (100, 100)).astype(np.uint8),
                raise_exceptions=True,
            )
        
        self.assertTrue(
            check_binary_image(
                img=np.random.randint(0, 2, (100, 100)).astype(np.bool),
                raise_exceptions=False,
                enforce_boolean=False
            )
        )

    def test_binary_or_3channel(self):
        """Test if the function decides between binary and 3-channel images correctly."""
        self.assertFalse(
            check_binary_image(
                img=np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8),
                raise_exceptions=False,
            )
        )
        with self.assertRaises(ImageNotBinaryError):
            check_binary_image(
                img=np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8),
                raise_exceptions=True,
            )


class TestLoadImage(TestCase):
    """Test the load_image function."""

    def setUp(self):
        super().setUp()
        self.base_path = os.path.join(os.path.dirname(__file__), "test_images", "tmp")
        self.test_grayscale_image_path = self.base_path + "/grayscale_image.png"
        self.test_rgb_image_path = self.base_path + "/rgb_image.png"
        self.test_bgr_image_path = self.base_path + "/bgr_image.png"

    def tearDown(self):
        test_files = [
            self.test_rgb_image_path,
            self.test_grayscale_image_path,
            self.test_bgr_image_path,
        ]
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)

    def test_load_existing_grayscale_image(self):
        """Test if the function loads an existing grayscale image."""
        self.test_grayscale_image = np.random.randint(0, 255, (100, 100))
        cv2.imwrite(self.test_grayscale_image_path, self.test_grayscale_image)
        self.assertTrue(os.path.exists(self.test_grayscale_image_path))

        grayscale_image = load_image(self.test_grayscale_image_path, mode="grayscale")
        self.assertTrue(np.array_equal(grayscale_image, self.test_grayscale_image))

    def test_load_existing_rgb_image(self):
        """Test if the function loads an existing RGB image."""
        self.test_rgb_image = np.random.randint(0, 255, (100, 100, 3))
        cv2.imwrite(self.test_rgb_image_path, self.test_rgb_image)
        self.assertTrue(os.path.exists(self.test_rgb_image_path))

        rgb_image = load_image(self.test_rgb_image_path, mode="RGB")
        self.assertTrue(np.array_equal(rgb_image, self.test_rgb_image))

    def test_load_existing_bgr_image(self):
        """Test if the function loads an existing BGR image."""
        self.test_bgr_image = np.random.randint(0, 255, (100, 100, 3))
        cv2.imwrite(self.test_rgb_image_path, self.test_bgr_image)
        self.assertTrue(os.path.exists(self.test_bgr_image_path))

        bgr_image = load_image(self.test_bgr_image_path, mode="BGR")
        self.assertTrue(np.array_equal(bgr_image, self.test_bgr_image_path))

    def test_load_non_existing_image(self):
        """Test if the function raises an error when loading a non-existing image."""
        with self.assertRaises(FileNotFoundError):
            load_image("non_existing_image.png", mode="RGB")


class TestGetBoundingRect(TestCase):

    def test_get_bounding_rect(self):
        """Test if the function returns the correct bounding rectangle."""
        self.fail("Not implemented yet.")


class TestRotateImage(TestCase):
    
    def test_rotate_image(self):
        """Test if the function rotates the image correctly."""
        self.fail("Not implemented yet.")


class TestRotatedRectWithMaxArea(TestCase):

    def test_rotated_rect_with_max_area(self):
        """Test if the function returns the correct rotated rectangle."""
        self.fail("Not implemented yet.")


class TestCropImage(TestCase):

    def test_crop_image(self):
        """Test if the function crops the image correctly."""
        self.fail("Not implemented yet.")
