import cv2
import matplotlib.pyplot as plt
import numpy as np
from .utils import check_rgb_image, show_images, check_grayscale_image
from .exceptions import ImageNotGrayscaleError, ImageNotRGBError


def remap_image_intensity(
    image: np.ndarray, range: tuple[int, int] = [0, 1], show_steps: bool = False
) -> np.ndarray:
    """Remap the intensity of the image to a specific range.

    Remap the intensity of the image to a specific range. This is useful to
    increase the contrast of the image while keeping relative intensities the same.

    Parameters
    ----------
    image : np.ndarray
        The image to remap the intensity of.
    range : tuple[int, int], optional
        The range to remap the intensity to. The default is [0,1].
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is False.

    Returns
    -------
    np.ndarray
        The image with the remapped intensity.

    Raises
    ------
    ImageNotGrayscaleError
        If the image is not grayscale.
    """
    if not isinstance(range, (tuple, list)):
        raise TypeError("Range should be a tuple not type {}".format(type(range)))
    if not isinstance(range[0], int) or not isinstance(range[1], int):
        raise TypeError(
            "Range should be a tuple of integers not type {} and {}".format(
                type(range[0]), type(range[1])
            )
        )
    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Image should be a numpy array not type {}".format(type(image))
        )

    # Only grayscale images can be remapped
    check_grayscale_image(image, raise_exceptions=True)

    # Remap the intensity of the image to the range
    remapped_image = cv2.normalize(image, None, range[0], range[1], cv2.NORM_MINMAX)

    if show_steps:
        show_images(
            {
                "Original image": image,
                "Remapped image": remapped_image,
                "Difference between the two images": image - remapped_image,
            }
        )

    return remapped_image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image intensity values to range [0, 255].

    This function is a wrapper around remap_image_intensity that normalizes the image to the range [0, 255].
    """
    return remap_image_intensity(image, (0, 255), show_steps=False)


def change_saturation(
    image: np.ndarray, delta: int, show_steps: bool = False
) -> np.ndarray:
    """Change the saturation of the image.

    Parameters
    ----------
    image : np.ndarray
        The image to change the saturation of.
    delta : int
        The amount to change the saturation. Positive values increase the saturation,
        negative values decrease the saturation.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is False.

    Returns
    -------
    np.ndarray
        The image with the changed saturation.

    Raises
    ------
    ImageNotRGBError
        If the image is not RGB.
    ValueError
        If the delta is not between -255 and 255.
    """
    if not isinstance(delta, int):
        raise TypeError("Delta should be an integer not type {}".format(type(delta)))
    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Image should be a numpy array not type {}".format(type(image))
        )

    check_rgb_image(image, raise_exceptions=True)

    if -255 > delta > 255:
        raise ValueError("Delta must be between -255 and 255!")

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Add the delta to the saturation channel
    if delta > 0:
        hsv[:, :, 1] += delta
    else:
        hsv[:, :, 1] -= abs(delta)

    # Convert back to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if show_steps:
        show_images(
            {
                "Original image": image,
                "Image with increased saturation": rgb,
                "Difference between the two images": image - rgb,
            }
        )

    return rgb


def change_brightness(
    image: np.ndarray, delta: int, show_steps: bool = False
) -> np.ndarray:
    """Change the brightness of the image.

    Parameters
    ----------
    image : np.ndarray
        The image to change the brightness of.
    delta : int
        The amount to change the brightness. Positive values increase the brightness,
        negative values decrease the brightness.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is False.

    Returns
    -------
    np.ndarray
        The image with the changed brightness.

    Raises
    ------
    ImageNotRGBError
        If the image is not RGB.
    ValueError
        If the delta is not between -255 and 255.
    """
    if not isinstance(delta, int):
        raise TypeError("Delta should be an integer not type {}".format(type(delta)))
    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Image should be a numpy array not type {}".format(type(image))
        )

    check_rgb_image(image, raise_exceptions=True)

    if -255 > delta > 255:
        raise ValueError("Delta must be between -255 and 255!")

    # Convert to HSV space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Add the delta to the value channel
    if delta > 0:
        hsv[:, :, 2] += delta
    else:
        hsv[:, :, 2] -= abs(delta)

    # Convert back to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if show_steps:
        show_images(
            {
                "Original image": image,
                "Image with increased brightness": rgb,
                "Difference between the two images": image - rgb,
            }
        )

    return rgb


def blur(image: np.ndarray, kernel_size: int, show_steps: bool = True) -> np.ndarray:
    """Blurs the image.

    The blurring is done using a simple average kernel.

    Parameters
    ----------
    image : np.ndarray
        The image to blur.
    kernel_size : int
        The size of the kernel to use for blurring.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is True.

    Returns
    -------
    np.ndarray
        The blurred image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Image should be a numpy array not type {}".format(type(image))
        )
    if not isinstance(kernel_size, int):
        raise TypeError(
            "kernel_size should be an integer not type {}".format(type(kernel_size))
        )
    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )

    # Create the kernel
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

    # Convolve the image with the kernel
    blurred_image = cv2.filter2D(image, -1, kernel)

    if show_steps:
        show_images(
            {
                "Original image": image,
                "Blurred image": blurred_image,
                "Difference between the two images": image - blurred_image,
            }
        )

    return blurred_image


def gaussian_blur(
    image: np.ndarray,
    kernel_size: int,
    sigma: int = 1,
    show_steps: bool = True,
) -> np.ndarray:
    """Blur the image using a gaussian filter

    Parameters
    ----------
    image : np.ndarray
        The image to blur
    kernel_size : int
        The kernel size for the gaussian filter
    sigma : int, optional
        The sigma value for the gaussian filter, by default 1
    show_steps : bool, optional
        If True, show the steps of the conversion, by default True

    Returns
    -------
    np.ndarray
        The blurred image
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array, not type {}".format(type(image)))
    if not isinstance(kernel_size, int):
        raise TypeError(
            "Gaussian kernel size must be an int, not type {}".format(type(kernel_size))
        )
    if not isinstance(sigma, int):
        raise TypeError("Gaus sigma must be an int, not type {}".format(type(sigma)))
    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps must be a boolean, not type {}".format(type(show_steps))
        )

    # Reduce noise using gaussian filter
    image = cv2.GaussianBlur(image, [kernel_size, kernel_size], sigma)

    if show_steps:
        plt.imshow(image)
        plt.title("Gaussian blur")
        plt.show()

    return image
