import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple, List
from .utils import check_rgb_image, crop_mask, show_images


def get_rgb_histogram(img: ArrayLike, show_steps: bool = False) -> ArrayLike:
    """Create a histogram of the image for each color channel.

    Parameters
    ----------
    img : ArrayLike
        The image to create the histogram of.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is False.

    Returns
    -------
    ArrayLike
        The histogram of the image.
    """
    if not isinstance(show_steps, bool):
        raise ValueError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(img, np.ndarray):
        raise ValueError(
            "Image should be a numpy array not type {}".format(type(img))
        )
    # Check if th image is BGR or RGB
    check_rgb_image(img, raise_exceptions=True)

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


def find_edges(image: ArrayLike, show_steps: bool = True) -> ArrayLike:
    """Detects edges in the image using kernel convolution.

    Parameters
    ----------
    image : ArrayLike
        The image to detect the edges in.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is True.

    Returns
    -------
    ArrayLike
        The edge detected image.
    """
    if not isinstance(show_steps, bool):
        raise ValueError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(image, np.ndarray):
        raise ValueError(
            "Image should be a numpy array not type {}".format(type(image))
        )
    # Create the kernel
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Convolve the image with the kernel
    edge_image = cv2.filter2D(image, -1, kernel)

    if show_steps:
        show_images({
            "Original image": image,
            "Edge detected image": edge_image
        })

    return edge_image


def find_components(
    image: ArrayLike, min_size: int, max_size: int, show_steps: bool = False
) -> Tuple[ArrayLike, List[ArrayLike], ArrayLike]:
    """Finds the connected components in the image.

    Parameters
    ----------
    image : ArrayLike
        The image to find the connected components in.
    min_size : int
        The minimum size of the connected components.
    max_size : int
        The maximum size of the connected components.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is False.

    Returns
    -------
    Tuple[ArrayLike, List[ArrayLike], ArrayLike]
        A tuple containing the image with the components marked by number,
        a list of the components and an image with all the components.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            "Image should be a numpy array not type {}".format(type(image))
        )
    if not isinstance(min_size, int):
        raise ValueError(
            "min_size should be an integer not type {}".format(type(min_size))
        )
    if not isinstance(max_size, int):
        raise ValueError(
            "max_size should be an integer not type {}".format(type(max_size))
        )
    if not isinstance(show_steps, bool):
        raise ValueError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    
    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        image, 8, cv2.CV_32S
    )

    # Create a smaller image for each component and a big image with all the components
    components = []
    total_components = np.zeros_like(image)
    for i in range(1, num_labels):
        if min_size < stats[i, cv2.CC_STAT_AREA] < max_size:
            # Create the full size mask, put the component in, crop away the
            # empty space and add it to the list
            component = np.zeros_like(image)
            component[labels == i] = 255
            total_components[labels == i] = 255
            component = crop_mask(component, show_steps=False)
            components.append(component)

    # Mark all the components by number in the original image
    for i, component in enumerate(components):
        image[component == 255] = i

    if show_steps:
        # Show the original image, the original with the components marked and the total components
        show_images({
            "Original image": image,
            "Image with components": image,
            "Total components": total_components
        })

    return image, components, total_components
