import cv2
import numpy as np
from typing import Tuple
from .utils import show_images
from .utils import check_grayscale_image
import copy


def threshold_percentage(
    image: np.ndarray,
    thresh_value: float = 0.8,
    invert: bool = False,
    show_steps: bool = False,
) -> np.ndarray:
    """Threshold using a percentage of the most bright pixel

    Parameters
    ----------
    image : np.ndarray
        The image to threshold
    thresh_value : float, optional
        The threshold limit (in % between 0-1), by default 0.8 times the most bright pixel
    invert : bool, optional
        Whether to invert the threshold, by default False
    show_steps : bool, optional
        Whether to show the steps, by default False

    Returns
    -------
    np.ndarray
        The thresholded image (a mask)
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array, not type {}".format(type(image)))
    if not isinstance(thresh_value, (float, int)):
        raise TypeError(
            "Thresh limit must be a float, not type {}".format(type(thresh_value))
        )
    if not isinstance(invert, bool):
        raise TypeError("Invert must be a bool, not type {}".format(type(invert)))
    if not isinstance(show_steps, bool):
        raise TypeError(
            "Show steps must be a bool, not type {}".format(type(show_steps))
        )
    if 0 > thresh_value > 1:
        raise ValueError(
            "Thresh limit must be between 0 and 1 not {}".format(thresh_value)
        )

    # Check if the image is grayscale
    check_grayscale_image(image, raise_exceptions=True)

    # Calculate the threshold and apply it
    thresh_value = thresh_value * np.max(image)
    _, mask = cv2.threshold(
        image, thresh_value, 255, cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    )

    if show_steps:
        show_images({"Original image": image, "Thresholded image": mask})

    return mask


def threshold_absolute(
    image: np.ndarray,
    thresh_value: int = 127,
    invert: bool = False,
    show_steps: bool = False,
) -> np.ndarray:
    """Threshold the image using an absolute value

    Parameters
    ----------
    image : np.ndarray
        The image to threshold
    thresh_value : int, optional
        The threshold value, by default 127
    invert : bool, optional
        Whether to invert the threshold, by default False
    show_steps : bool, optional
        Whether to show the steps, by default False

    Returns
    -------
    np.ndarray
        The thresholded image
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array, not type {}".format(type(image)))
    if not isinstance(thresh_value, int):
        raise TypeError(
            "Thresh value must be an int, not type {}".format(type(thresh_value))
        )
    if not isinstance(invert, bool):
        raise TypeError("Invert must be a bool, not type {}".format(type(invert)))
    if not isinstance(show_steps, bool):
        raise TypeError(
            "Show steps must be a bool, not type {}".format(type(show_steps))
        )
    # Check if the image is grayscale
    check_grayscale_image(image, raise_exceptions=True)

    # Apply the threshold
    _, mask = cv2.threshold(
        image, thresh_value, 255, cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    )

    if show_steps:
        show_images({"Original image": image, "Thresholded image": mask})

    return mask


def adaptive_threshold(
    image: np.ndarray,
    correction: int = 5,
    kernel_size: int = 21,
    invert: bool = False,
    show_steps: bool = False,
) -> np.ndarray:
    """Apply an adaptive threshold to the image

    Parameters
    ----------
    image : np.ndarray
        The image to threshold
    correction : int, optional
        The correction value for the adaptive threshold, by default 5
    kernel_size : int, optional
        The kernel size for the adaptive threshold, by default 21
    invert : bool, optional
        Whether to invert the threshold, by default False
    show_steps : bool, optional
        Whether to show the steps, by default False

    Returns
    -------
    np.ndarray
        The thresholded image
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array, not type {}".format(type(image)))
    if not isinstance(correction, int):
        raise TypeError(
            "Correction must be an int, not type {}".format(type(correction))
        )
    if not isinstance(kernel_size, int):
        raise TypeError(
            "Kernel size must be an int, not type {}".format(type(kernel_size))
        )
    if not isinstance(invert, bool):
        raise TypeError("Invert must be a bool, not type {}".format(type(invert)))
    if not isinstance(show_steps, bool):
        raise TypeError(
            "Show steps must be a bool, not type {}".format(type(show_steps))
        )
    # Check if the image is grayscale
    check_grayscale_image(image, raise_exceptions=True)

    # Apply the adaptive threshold
    image = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
        kernel_size,
        correction,
    )

    if show_steps:
        show_images({"Original image": image, "Thresholded image": image})

    return image


def hsv_color_filter(
    image: np.ndarray,
    lower_bound: Tuple[int, int, int] = (0, 0, 0),
    upper_bound: Tuple[int, int, int] = (0, 0, 0),
    show_steps: bool = False,
) -> np.ndarray:
    """Filter the image based on a color range in HSV

    .. attention::
        The provided image must already be in HSV format.

    Parameters
    ----------
    image : np.ndarray
        The image to filter. Expects a HSV image
    color_range : tuple[tuple[int, int, int], tuple[int, int, int]], optional
        The color range to filter, by default ((0, 0, 0,), (0, 0, 0,)). Because
        this function uses opencv the ranges are:
        * Hue: [0, 179]
        * Saturation: [0, 255]
        * Value: [0, 255]
    show_steps : bool, optional
        Whether to show the steps, by default False

    Returns
    -------
    np.ndarray
        The filtered image
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array, not type {}".format(type(image)))
    if not isinstance(lower_bound, tuple) or not isinstance(upper_bound, tuple):
        raise TypeError(
            "Lower and upper bounds must be tuples, not type {} and {}".format(
                type(lower_bound), type(upper_bound)
            )
        )
    if not len(lower_bound) == 3 or not len(upper_bound) == 3:
        raise ValueError(
            "Lower and upper bounds must have 3 values, not {}".format(
                len(lower_bound), len(upper_bound)
            )
        )
    for value in lower_bound + upper_bound:
        if not 0 <= value <= 255:
            raise ValueError(
                "All values in the color range must be between 0 and 255, not {}".format(
                    value
                )
            )
    if not isinstance(show_steps, bool):
        raise TypeError(
            "Show steps must be a bool, not type {}".format(type(show_steps))
        )
    # Filter the color range
    # converted_stack = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    converted_stack = copy.deepcopy(image)
    mask = cv2.inRange(converted_stack, lower_bound, upper_bound)
    # Remove the colors that did not pass the mask
    filtered = np.ones_like(image) * 255
    filtered[mask == 0] = 0

    if show_steps:
        show_images({"Original image": image, "Filtered image": filtered})
    return filtered
