import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from scipy import fftpack
from scipy.signal import find_peaks
from matplotlib.colors import LogNorm
from .utils import check_rgb_image, show_images


def rgb_to_grayscale(img: np.ndarray, show_steps: bool = False) -> np.ndarray:
    """Convert an RGB image to a grayscale image.

    Parameters
    ----------
    img : np.ndarray
        The image to convert to grayscale.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is False.

    Returns
    -------
    np.ndarray
        The grayscale image.
    """
    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(img, np.ndarray):
        raise TypeError(
            "Image should be a numpy array not type {}".format(type(img))
        )
    check_rgb_image(img, raise_exceptions=True)

    # Convert the image to grayscale
    gray_img = np.mean(img, axis=2)

    if show_steps:
        show_images({
            "Original image": img,
            "Grayscale image": gray_img
        })

    return gray_img


def grayscale_to_binary(img: np.ndarray, show_steps: bool = False) -> np.ndarray:
    """Convert a grayscale image to a binary image.

    Parameters
    ----------
    img : np.ndarray
        The image to convert to binary.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is False.

    Returns
    -------
    np.ndarray
        The binary image.
    """
    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(img, np.ndarray):
        raise TypeError(
            "Image should be a numpy array not type {}".format(type(img))
        )
    
    if len(img.shape) == 3:
        raise TypeError("Image must be grayscale!")

    # Convert to binary image
    binary_img = img > 127

    if show_steps:
        show_images({
            "Original image": img,
            "Binary image": binary_img
        })

    return binary_img


def rgb_to_binary(img: np.ndarray, show_steps: bool = False) -> np.ndarray:
    """Convert an RGB image to a binary image.

    Parameters
    ----------
    img : np.ndarray
        The image to convert to binary.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is False.

    Returns
    -------
    np.ndarray
        The binary image.
    """
    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(img, np.ndarray):
        raise TypeError(
            "Image should be a numpy array not type {}".format(type(img))
        )
    
    check_rgb_image(img, raise_exceptions=True)
    gray_img = rgb_to_grayscale(img, show_steps=False)
    binary_img = grayscale_to_binary(gray_img, show_steps=False)

    if show_steps:
        show_images({
            "Original image": img,
            "Binary image": binary_img
        })

    return binary_img


def convert_to_fft_image(image: np.ndarray, show_steps: bool = True) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """
    Converts the image to a fourrier transformed image.

    .. attention::
        This function does accept color images, but will convert them to grayscale.

    The function will generate the fft image and try to estimate the dominant frequency and angle.

    Parameters
    ----------
    image : np.ndarray
        The image to convert to a fourrier transformed image.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int, float]
        A tuple containing the fourrier transformed image, the 1D spectrum, the dominant 
        frequency and the angle of the dominant frequency.
    """
    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Image should be a numpy array not type {}".format(type(image))
        )
    
    # If not grayscale, convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Calculate the x and y component fft
    fft_img = fftpack.fftshift(fftpack.fft2(image))
    fft_img = np.abs(fft_img)

    # Use a hamming window on the image to correct for the edge effects
    fft_img = fft_img * np.hamming(fft_img.shape[0])[:, np.newaxis]

    # Multiply the x component with the y component to get a 1D spectrum
    fft1d = fft_img[:, 0] * fft_img[0, :]
    fft1d = fft1d[: fft1d.shape[0] // 2]

    # Find the dominant frequency
    peaks, _ = find_peaks(fft1d, prominence=1)
    wavenumber = np.argmax(fft1d[peaks])

    # Determine the angle of the dominant frequency
    angle = np.arctan(
        (fft_img.shape[0] // 2 - wavenumber) / (wavenumber - fft_img.shape[0] // 2)
    )

    if show_steps:
        print(f"Angle: {angle} rad")
        print(f"Angle: {np.degrees(angle)} deg")
        print(f"Wavenumber: {wavenumber} 1/px")
        show_images({
            "Original image": image,
            "Fourrier transformed image": fft_img
        })

    return fft_img, fft1d, wavenumber, angle
