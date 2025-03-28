import cv2
import numpy as np
from typing import Tuple
from scipy import fftpack
from scipy.signal import find_peaks
from .utils import check_three_channel_image, show_images, check_grayscale_image


def bgr_to_grayscale(image: np.ndarray, show_steps: bool = True) -> np.ndarray:
    """Converts an RGB image to a grayscale image.

    .. attention::
        The image is assumed to be in BGR format but any other 3 channel image
        will be accepted as well. This can lead to unexpected results.


    Parameters
    ----------
    image : np.ndarray
        The image to convert to a grayscale image.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is True.

    Returns
    -------
    np.ndarray
        The grayscale image.

    Raises
    ------
    ImageNot3ChannelError
        If the image is not a 3 channel image
    """
    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(image, np.ndarray):
        raise TypeError("Image should be a numpy array not type {}".format(type(image)))

    check_three_channel_image(image, raise_exceptions=True)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if show_steps:
        show_images({"Original image": image, "Grayscale image": gray_image})

    return gray_image


def bgr_to_hsv(image: np.ndarray, show_steps: bool = True) -> np.ndarray:
    """
    Converts an BGR image to an HSV image.

    .. attention::
        The image is assumed to be in BGR format but any other 3 channel image
        will be accepted as well. This can lead to unexpected results

    Parameters
    ----------
    image : np.ndarray
        The image to convert to an HSV image.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is True.

    Returns
    -------
    np.ndarray
        The HSV image.

    Raises
    ------
    ImageNot3ChannelError
        If the image is not a 3 channel image
    """
    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(image, np.ndarray):
        raise TypeError("Image should be a numpy array not type {}".format(type(image)))

    check_three_channel_image(image, raise_exceptions=True)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if show_steps:
        show_images({"Original image": image, "HSV image": hsv_image})

    return hsv_image


def hsv_to_bgr(image: np.ndarray, show_steps: bool = True) -> np.ndarray:
    """Coverts an HSV image to a BGR image.

    .. attention::
        The image is assumed to be in HSV format but any other 3 channel image
        will be accepted as well. This can lead to unexpected results

    Parameters
    ----------
    image : np.ndarray
        The image to convert to a BGR image.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is True.

    Returns
    -------
    np.ndarray
        The BGR image.

    Raises
    ------
    ImageNot3ChannelError
        If the image is not a 3 channel image
    """
    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(image, np.ndarray):
        raise TypeError("Image should be a numpy array not type {}".format(type(image)))

    check_three_channel_image(image, raise_exceptions=True)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    if show_steps:
        show_images({"Original image": image, "BGR image": bgr_image})

    return bgr_image


def bgr_to_rgb(image: np.ndarray, show_steps: bool = True) -> np.ndarray:
    """
    Converts a BGR image to an RGB image.

    Mosly used for visualization purposes.

    .. attention::
        The image is assumed to be in BGR format but any other 3 channel image
        will be accepted as well. This can lead to unexpected results

    Parameters
    ----------
    image : np.ndarray
        The image to convert to an RGB image.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is True.

    Returns
    -------
    np.ndarray
        The RGB image.
    """
    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(image, np.ndarray):
        raise TypeError("Image should be a numpy array not type {}".format(type(image)))

    check_three_channel_image(image, raise_exceptions=True)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if show_steps:
        show_images({"Original image": image, "RGB image": rgb_image})

    return rgb_image


def invert_image(image: np.ndarray) -> np.ndarray:
    """Invert the image.

    Parameters
    ----------
    image : np.ndarray
        The image to invert.

    Returns
    -------
    np.ndarray
        The inverted image.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            "image should be a numpy array not type {}".format(type(image))
        )

    if len(image.shape) == 2:
        return 255 - image
    else:
        return cv2.bitwise_not(image)


def grayscale_to_fft_image(
    image: np.ndarray, show_steps: bool = True
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """
    Converts the image to a fourrier transformed image.

    The function will generate the fft image and try to estimate the dominant frequency and angle.

    .. attention::
        This function is broken for now and will be fixed somewhere in the future, as I do
        not have the time to fix it right now.

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

    Raises
    ------
    ImageNotGrayscaleError
        If the image is not a grayscale
    """
    raise NotImplementedError(
        "This function is broken and will be fixed somewhere in the future"
    )

    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(image, np.ndarray):
        raise TypeError("Image should be a numpy array not type {}".format(type(image)))

    check_grayscale_image(image, raise_exceptions=True)

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
        show_images({"Original image": image, "Fourrier transformed image": fft_img})

    return fft_img, fft1d, wavenumber, angle
