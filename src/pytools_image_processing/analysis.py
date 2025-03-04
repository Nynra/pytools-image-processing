import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from .utils import (
    check_three_channel_image,
    check_grayscale_image,
)


def plot_intensity_profile(
    image: np.ndarray, show_steps: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Plot the intensity profile of the image.

    This function plots the intensity profile of the image. This can be used to
    find the edges of the image.

    Parameters
    ----------
    image : np.ndarray
        The image to plot the intensity profile of.
    show_steps : bool, optional
        If True show the resulting plot. The default is False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The x and y arrays of the intensity mesh grid.

    Raises
    ------
    ImageNotGrayscaleError
        If the image is not a grayscale image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image should be a numpy array not type {}".format(type(image)))
    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    # Check if the image is grayscale
    check_grayscale_image(image, raise_exceptions=True)

    # Create a mesh grid and plot the inverse intensity of the object
    # Change the only the black spots to white
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

    if show_steps:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, image, cmap="viridis")
        plt.show()

    return x, y


def get_rgb_histogram(img: np.ndarray, show_steps: bool = False) -> np.ndarray:
    """Create a histogram of the image for each color channel.

    This function creates a histogram of the image for each color channel. This
    can be used to find the distribution of the colors in the image.

    .. attention::
        The image is assumed to be in RGB format but any other 3 channel image
        will be accepted as well. This can lead to unexpected results.

    Parameters
    ----------
    img : np.ndarray
        The image to create the histogram of.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is False.

    Returns
    -------
    np.ndarray
        The histogram of the image.

    Raises
    ------
    ImageNot3ChannelError
        If the image is not a 3 channel image.
    """
    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(img, np.ndarray):
        raise TypeError("Image should be a numpy array not type {}".format(type(img)))
    # Check if th image is BGR or RGB
    check_three_channel_image(img, raise_exceptions=True)

    # Create a histogram for each channel
    r_hist = np.histogram(img[:, :, 0], bins=256, range=(0, 255))[0]
    g_hist = np.histogram(img[:, :, 1], bins=256, range=(0, 255))[0]
    b_hist = np.histogram(img[:, :, 2], bins=256, range=(0, 255))[0]

    # Combine the histograms
    rgb_hist = np.stack([r_hist, g_hist, b_hist], axis=1)

    if show_steps:
        # Plot the image
        plt.figure(figsize=(20, 10))
        plt.subplot(131)
        plt.imshow(img)

        # Plot the seperate colors in one histogram
        plt.subplot(132)
        plt.bar(np.arange(256), r_hist, color="r", alpha=0.5, label="Red")
        plt.bar(np.arange(256), g_hist, color="g", alpha=0.5, label="Green")
        plt.bar(np.arange(256), b_hist, color="b", alpha=0.5, label="Blue")
        plt.title("Histogram of the seperate colors")
        plt.legend()

        # Also plot the sum of the histograms in a seperate plot
        plt.subplot(133)
        plt.bar(np.arange(256), np.sum(rgb_hist, axis=1), color="k", alpha=0.5)
        plt.title("Sum of the histograms")
        plt.show()

    return rgb_hist
