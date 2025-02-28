import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Tuple
from .exceptions import ImageNotBinaryError, ImageNotGrayscaleError, ImageNot3ChannelError


def show_images(images: dict[str, np.ndarray], show_axis: bool = True) -> ...:
    """Show the images in a grid.

    Plot the given images as subplots in a grid.

    Parameters
    ----------
    images : dict[str, np.ndarray]
        The images to show.
    show_axis : bool, optional
        If True, show the axis of the images. The default is True.
    """
    if not isinstance(images, dict):
        raise TypeError(
            "images should be a dictionary not type {}".format(type(images))
        )
    if not all(isinstance(k, str) for k in images.keys()):
        raise TypeError("Keys of images should be strings")
    if not all(isinstance(v, np.ndarray) for v in images.values()):
        raise TypeError("Values of images should be numpy arrays")
    if not isinstance(show_axis, bool):
        raise TypeError(
            "show_axis should be a boolean not type {}".format(type(show_axis))
        )

    # Try to make the grid as square as possible
    n = len(images)
    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))

    # Create the grid
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot the images
    for i, (title, img) in enumerate(images.items()):
        # Correct the color of the image
        if len(img.shape) == 2:
            axes[i].imshow(img, cmap="gray")
        else:
            axes[i].imshow(img)

        axes[i].set_title(title)
        if not show_axis:
            axes[i].axis("off")

    # Remove the empty subplots
    for i in range(n, rows * cols):
        fig.delaxes(axes[i])

    plt.show()


def check_three_channel_image(img: np.ndarray, raise_exceptions: bool = True) -> bool:
    """Check if the image is a three channel image.

    Check if the image is a three channel image, for example RGB or BGR.

    Parameters
    ----------
    img : np.ndarray
        The image to check.
    raise_exceptions : bool, optional
        If True, raise an exception if the image is not RGB. The default is True.

    Returns
    -------
    bool
        True if the image is RGB, False otherwise.

    Raises
    ------
    ImageNotRGBError
        If the image is not RGB and raise_exceptions is True.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("Image should be a numpy array not type {}".format(type(img)))
    if not isinstance(raise_exceptions, bool):
        raise TypeError(
            "raise_exceptions should be a boolean not type {}".format(
                type(raise_exceptions)
            )
        )
    # Check if the image is rgb or not
    try:
        if img.shape[2] == 3:
            return True
    except IndexError:
        pass

    if raise_exceptions:
        raise ImageNot3ChannelError("Image is not a 3 channel image!")
    return False


def check_grayscale_image(img: np.ndarray, raise_exceptions: bool = True, enforce_not_boolean:bool=True) -> bool:
    """Check if the image is a grayscale image.

    Parameters
    ----------
    img : np.ndarray
        The image to check.
    raise_exceptions : bool, optional
        If True, raise an exception if the image is not grayscale. The default is True.
    enforce_not_boolean : bool, optional
        If True, enforce that the image is not a boolean image, otherwise it will be
        considered grayscale. The default is True.

    Returns
    -------
    bool
        True if the image is grayscale, False otherwise.

    Raises
    ------
    ImageNotGrayscaleError
        If the image is not grayscale and raise_exceptions is True.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("Image should be a numpy array not type {}".format(type(img)))
    if not isinstance(raise_exceptions, bool):
        raise TypeError(
            "raise_exceptions should be a boolean not type {}".format(
                type(raise_exceptions)
            )
        )
    if not isinstance(enforce_not_boolean, bool):
        raise TypeError(
            "enforce_not_boolean should be a boolean not type {}".format(
                type(enforce_not_boolean)
            )
        )
    # Check if the image is grayscale or not
    if len(img.shape) == 2:
        if enforce_not_boolean and img.dtype == np.bool:
            if raise_exceptions:
                raise ImageNotGrayscaleError("Image is not grayscale!")
            else:
                return False
        return True

    if raise_exceptions:
        raise ImageNotGrayscaleError("Image is not grayscale!")
    else:
        return False


def check_binary_image(img: np.ndarray, raise_exceptions: bool = True, enforce_boolean:bool=True) -> bool:
    """Check if the image is a binary image.

    Parameters
    ----------
    img : np.ndarray
        The image to check.
    raise_exceptions : bool, optional
        If True, raise an exception if the image is not binary. The default is True.
    enforce_boolean : bool, optional
        If True, enforce that the image is a boolean image, any 2D image with only 2 unique
        values will be considered binary. The default is True.

    Returns
    -------
    bool
        True if the image is binary, False otherwise.

    Raises
    ------
    ImageNotBinaryError
        If the image is not binary and raise_exceptions is True.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("Image should be a numpy array not type {}".format(type(img)))
    if not isinstance(raise_exceptions, bool):
        raise TypeError(
            "raise_exceptions should be a boolean not type {}".format(
                type(raise_exceptions)
            )
        )
    if not isinstance(enforce_boolean, bool):
        raise TypeError(
            "enforce_boolean should be a boolean not type {}".format(
                type(enforce_boolean)
            )
        )
    # Check if the image is binary or not
    if len(img.shape) != 2:
        if raise_exceptions:
            raise ImageNotBinaryError("Image is not binary!")
        else:
            return False
    elif img.dtype == np.bool:
        # Image is 2D with only boolean values
        return True
    elif len(np.unique(img)) == 2 and not enforce_boolean:
        # Image is 2D with 2 unique values (can be 0-1 but also 0-255, 40-70, etc.)
        return True
    else:
        if raise_exceptions:
            raise ImageNotBinaryError("Image is not binary!")
        else:
            return False


def load_image(
    filename: str,
    mode: str = "bgr",
    in_file_dir: bool = False,
) -> np.ndarray:
    """Load the image using opencv.

    Parameters
    ----------
    filename : str
        The filename of the image to load.
    mode : str, optional
        The mode of the image. The options are:
            - "rgb": Load the image as RGB.
            - "bgr": Load the image as BGR.
            - "grayscale": Load the image as grayscale.
        The default is "bgr".
    in_file_dir : bool, optional
        If True, the filename is relative to the current working directory.
        The default is False.

    Returns
    -------
    np.ndarray
        The image.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not isinstance(filename, str):
        raise TypeError(
            "filename should be a string not type {}".format(type(filename))
        )
    if not isinstance(mode, str):
        raise TypeError("mode should be a string not type {}".format(type(mode)))
    if mode.lower() not in ("rgb", "bgr", "grayscale"):
        raise TypeError(
            "mode should be 'rgb', 'bgr' or 'grayscale' not {}".format(mode)
        )
    if not isinstance(in_file_dir, bool):
        raise TypeError(
            "in_file_dir should be a boolean not type {}".format(type(in_file_dir))
        )

    # Check if the file exists
    if in_file_dir:
        filename = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File {filename} does not exist!")

    # Convert the image to the right mode
    match mode.lower():
        case "rgb":
            img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        case "bgr":
            img = cv2.imread(filename)
        case "grayscale":
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        case _:
            raise ValueError(f"Mode {mode} is not supported!")

    return img


def save_image(
    filename: str,
    mode: str,
    image: np.ndarray,
    in_file_dir: bool = False,
) -> ...:
    """Save the image using opencv.

    Parameters
    ----------
    filename : str
        The filename of the image to save.
    mode : str
        The mode of the image. The options are:
            - "rgb": Save the image as RGB.
            - "bgr": Save the image as BGR.
            - "grayscale": Save the image as grayscale.
        The default is "bgr".
    image : np.ndarray
        The image to save.
    in_file_dir : bool, optional
        If True, the filename is relative to the current working directory.
        The default is False.

    Raises
    ------
    ValueError
        If the mode is not supported.
    """
    if not isinstance(filename, str):
        raise TypeError(
            "filename should be a string not type {}".format(type(filename))
        )
    if not isinstance(mode, str):
        raise TypeError("mode should be a string not type {}".format(type(mode)))
    if mode.lower() not in ("rgb", "bgr", "grayscale"):
        raise TypeError(
            "mode should be 'rgb', 'bgr' or 'grayscale' not {}".format(mode)
        )
    if not isinstance(in_file_dir, bool):
        raise TypeError(
            "in_file_dir should be a boolean not type {}".format(type(in_file_dir))
        )

    # Convert the image to the right mode
    match mode.lower():
        case "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        case "bgr" | "grayscale":
            pass
        case _:
            raise ValueError(f"Mode {mode} is not supported!")

    # Save the image
    if in_file_dir:
        filename = os.path.join(os.path.dirname(__file__), filename)
    cv2.imwrite(filename, image)

