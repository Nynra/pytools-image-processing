import cv2
import numpy as np
from typing import List, Tuple
from .utils import show_images
from .utils import check_grayscale_image


def threshold(
    image: np.ndarray,
    thresh_limit: float = 0.8,
    threshold_kernel_size: int = 21,
    show_steps: bool = False,
) -> np.ndarray:
    """Threshold the image

    Parameters
    ----------
    image : np.ndarray
        The image to threshold
    thresh_limit : float, optional
        The threshold limit, by default 0.8 times the most bright pixel
    threshold_kernel_size : int, optional
        The kernel size for the threshold, by default 21
    show_steps : bool, optional
        Whether to show the steps, by default False

    Returns
    -------
    np.ndarray
        The thresholded image
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Image must be a numpy array, not type {}".format(type(image))
        )
    if not isinstance(thresh_limit, (float, int)):
        raise TypeError(
            "Thresh limit must be a float, not type {}".format(type(thresh_limit))
        )
    if not isinstance(threshold_kernel_size, int):
        raise TypeError(
            "Threshold kernel size must be an int, not type {}".format(
                type(threshold_kernel_size)
            )
        )

    # Calculate the threshold
    thresh_limit = thresh_limit * np.max(image)
    # Apply the threshold
    _, mask = cv2.threshold(image, thresh_limit, 255, cv2.THRESH_BINARY)

    if show_steps:
        show_images({"Original image": image, "Thresholded image": mask})

    return mask


def adaptive_threshold(
    image: np.ndarray,
    correction: int = 5,
    kernel_size: int = 21,
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
    show_steps : bool, optional
        Whether to show the steps, by default False

    Returns
    -------
    np.ndarray
        The thresholded image
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Image must be a numpy array, not type {}".format(type(image))
        )
    if not isinstance(correction, int):
        raise TypeError(
            "Correction must be an int, not type {}".format(type(correction))
        )
    if not isinstance(kernel_size, int):
        raise TypeError(
            "Kernel size must be an int, not type {}".format(type(kernel_size))
        )
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
        cv2.THRESH_BINARY,
        kernel_size,
        correction,
    )

    if show_steps:
        show_images({"Original image": image, "Thresholded image": image})

    return image


def color_filter(
    image: np.ndarray,
    color_range=((0,0,0,),(0,0,0,),),  # In HSV
    show_steps: bool = False,
    ) -> np.ndarray:
        """Filter the image based on color

        Parameters
        ----------
        image : np.ndarray
            The image to filter. Expects a BGR image
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
        if not isinstance(color_range, tuple):
            raise TypeError(
                "Color range must be a tuple, not type {}".format(type(color_range))
            )
        if not isinstance(color_range[0], tuple) or not isinstance(
            color_range[1], tuple
        ):
            raise TypeError(
                "Color range must be a tuple of tuples, not type {} and {}".format(
                    type(color_range[0]), type(color_range[1])
                )
            )
        if len(color_range[0]) != 3 or len(color_range[1]) != 3:
            raise TypeError(
                "Color range must be a tuple of tuples of length 3, not length {} and {}".format(
                    len(color_range[0]), len(color_range[1])
                )
            )

        # Check if upper is lower than lower and everything between 0 and 255
        for i in range(3):
            if color_range[0][i] > color_range[1][i]:
                raise ValueError(
                    "Lower bound is higher than upper bound for color filter"
                )
            if color_range[0][i] < 0 or color_range[0][i] > 255:
                raise ValueError("Lower bound is not between 0 and 255")
            if color_range[1][i] < 0 or color_range[1][i] > 255:
                raise ValueError("Upper bound is not between 0 and 255")

        # Filter the color range
        converted_stack = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(converted_stack, color_range[0], color_range[1])
        # Remove the colors that did not pass the mask
        filtered = np.ones_like(image) * 255
        filtered[mask == 0] = 0

        if show_steps:
            show_images({"Original image": image, "Filtered image": filtered})
        return filtered