import numpy as np
from .morphologic import (
    remap_image_intensity,
    normalize_image,
    change_brightness,
    change_saturation,
    average_blur,
    gaussian_blur,
)
from .filtering import (
    threshold_absolute,
    threshold_percentage,
    adaptive_threshold,
    hsv_color_filter,
)
from .conversions import bgr_to_grayscale, bgr_to_rgb


class ImageAnalyzer:
    """A simple class that holds an image and provides methods for image process.

    This class is a convenient way of using the image processing functions from the pytools_image_processing package.
    """

    _og_image: np.ndarray
    _image: np.ndarray

    def __init__(self, image: np.ndarray):
        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Image should be a numpy array not type {}".format(type(image))
            )
        self._og_image = image
        self._image = image

    # Morphologic methods
    def remap_image_intensity(self, min: int, max: int) -> np.ndarray:
        self._image = remap_image_intensity(image=self._image, range=(min, max))

    def normalize_image(self) -> np.ndarray:
        self._image = normalize_image(image=self._image)

    def change_saturation(self, value: int) -> np.ndarray:
        self._image = change_saturation(image=self._image, value=value)

    def change_brightness(self, value: int) -> np.ndarray:
        self._image = change_brightness(image=self._image, value=value)

    def average_blur(self, kernel_size: int) -> np.ndarray:
        self._image = average_blur(image=self._image, kernel_size=kernel_size)

    def gaussian_blur(self, kernel_size: int, sigma: int) -> np.ndarray:
        self._image = gaussian_blur(
            image=self._image, kernel_size=kernel_size, sigma=sigma
        )

    # Filtering methods
    def threshold_absolute(self, thresh_value: int, invert: bool) -> np.ndarray:
        self._image = threshold_absolute(
            image=self._image, thresh_value=thresh_value, invert=invert
        )

    def threshold_percentage(self, percentage: int, invert: bool) -> np.ndarray:
        self._image = threshold_percentage(
            image=self._image, percentage=percentage, invert=invert
        )

    def adaptive_threshold(self, correction: int, kernel_size: int) -> np.ndarray:
        self._image = adaptive_threshold(
            image=self._image, correction=correction, kernel_size=kernel_size
        )

    def hsv_color_filter(
        self, lower_bound: np.ndarray, upper_bound: np.ndarray
    ) -> np.ndarray:
        self._image = hsv_color_filter(
            image=self._image, lower_bound=lower_bound, upper_bound=upper_bound
        )

    # Conversion methods
    def bgr_to_grayscale(self) -> np.ndarray:
        self._image = bgr_to_grayscale(image=self._image)

    def bgr_to_rgb(self) -> np.ndarray:
        self._image = bgr_to_rgb(image=self._image)
