import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import os
from typing import Tuple



def show_images(images: dict[str, ArrayLike]):
    """Show the images in a grid.

    Plot the given images as subplots in a grid.

    Parameters
    ----------
    images : dict[str, ArrayLike]
        The images to show.
    """
    if not isinstance(images, dict):
        raise ValueError(
            "images should be a dictionary not type {}".format(type(images))
        )
    if not all(isinstance(k, str) for k in images.keys()):
        raise ValueError("Keys of images should be strings")
    if not all(isinstance(v, np.ndarray) for v in images.values()):
        raise ValueError("Values of images should be numpy arrays")

    # Try to make the grid as square as possible
    n = len(images)
    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))

    # Create the grid
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()

    # Plot the images
    for i, (title, img) in enumerate(images.items()):
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis("off")

    # Remove the empty subplots
    for i in range(n, rows * cols):
        fig.delaxes(axes[i])

    plt.show()


def check_rgb_image(img: ArrayLike, raise_exceptions: bool = True) -> bool:
    """Check if the image is an RGB image.

    Parameters
    ----------
    img : ArrayLike
        The image to check.
    raise_exceptions : bool, optional
        If True, raise an exception if the image is not RGB. The default is True.

    Returns
    -------
    bool
        True if the image is RGB, False otherwise.

    Raises
    ------
    ValueError
        If the image is not RGB and raise_exceptions is True.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("Image should be a numpy array not type {}".format(type(img)))
    if not isinstance(raise_exceptions, bool):
        raise ValueError(
            "raise_exceptions should be a boolean not type {}".format(
                type(raise_exceptions)
            )
        )
    # Check if the image is rgb or not
    if img.shape[2] == 3:
        return True
    else:
        if raise_exceptions:
            raise ValueError("Image is not RGB!")
        else:
            return False


def load_image(
    filename: str, in_file_dir: bool = True, show_steps: bool = False
) -> ArrayLike:
    """Load the image using opencv.

    The image is loaded using opencv and converted to RGB.

    Parameters
    ----------
    filename : str
        The filename of the image to load.
    in_file_dir : bool, optional
        If True, the filename is relative to the file directory. The default is True.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is False.

    Returns
    -------
    ArrayLike
        The image.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not isinstance(filename, str):
        raise ValueError(
            "filename should be a string not type {}".format(type(filename))
        )
    if not isinstance(in_file_dir, bool):
        raise ValueError(
            "in_file_dir should be a boolean not type {}".format(type(in_file_dir))
        )
    if not isinstance(show_steps, bool):
        raise ValueError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )

    # Check if the file exists
    if in_file_dir:
        filename = os.path.join(os.path.dirname(__file__), filename)

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File {filename} does not exist!")

    img = cv2.imread(
        filename,
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if show_steps:
        show_images({"Original image": img})

    return img


def crop_mask(mask: ArrayLike, show_steps: bool = True) -> ArrayLike:
    """Crops the mask to the smallest possible size.

    Parameters
    ----------
    mask : ArrayLike
        The mask to crop.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is True.

    Returns
    -------
    ArrayLike
        The cropped mask.
    """
    if not isinstance(mask, np.ndarray):
        raise ValueError("mask should be a numpy array not type {}".format(type(mask)))
    if not isinstance(show_steps, bool):
        raise ValueError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )

    # Find the bounding box of the mask
    x, y, w, h = cv2.boundingRect(mask)

    # Crop the mask
    cropped_mask = mask[y : y + h, x : x + w]

    if show_steps:
        show_images({"Original mask": mask, "Cropped mask": cropped_mask})

    return cropped_mask

