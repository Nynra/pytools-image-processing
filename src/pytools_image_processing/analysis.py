import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from .utils import (
    check_rgb_image,
    show_images,
    check_binary_image,
    check_grayscale_image,
)
import copy


def plot_intensity_profile(image: np.ndarray) -> np.ndarray:
    """Plot the intensity profile of the image.

    This function plots the intensity profile of the image. This can be used to
    find the edges of the image.

    Parameters
    ----------
    image : np.ndarray
        The image to plot the intensity profile of.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            "Image should be a numpy array not type {}".format(type(image))
        )
    # Check if the image is grayscale
    check_grayscale_image(image, raise_exceptions=True)

    # Create a mesh grid and plot the inverse intensity of the object
    # Change the only the black spots to white
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, object, cmap="viridis")
    plt.show()


def get_rgb_histogram(img: np.ndarray, show_steps: bool = False) -> np.ndarray:
    """Create a histogram of the image for each color channel.

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
    """
    if not isinstance(show_steps, bool):
        raise ValueError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(img, np.ndarray):
        raise ValueError("Image should be a numpy array not type {}".format(type(img)))
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


def find_edges(image: np.ndarray, show_steps: bool = True) -> np.ndarray:
    """Detects edges in the grayscale image using kernel convolution.

    This function uses a kernel to detect edges in the image. The kernel is
    convolved with the image to detect the edges.

    Parameters
    ----------
    image : np.ndarray
        The image to detect the edges in.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is True.

    Returns
    -------
    np.ndarray
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
    # Check if the image is grayscale
    check_grayscale_image(image, raise_exceptions=True)

    # Create the kernel
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Convolve the image with the kernel
    edge_image = cv2.filter2D(image, -1, kernel)

    if show_steps:
        show_images({"Original image": image, "Edge detected image": edge_image})

    return edge_image


def find_components(
    image: np.ndarray,
    min_size: int,
    max_size: int,
    crop_components: bool = False,
    show_steps: bool = False,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Finds the connected components in the binary image.

    This function uses the :func:`cv2.connectedComponentsWithStats` function to find
    the connected components in a binary image. It then creates a smaller
    image for each component and a big image with all the components.

    .. attention::

        This function assumes that the input image is a binary image.

    Parameters
    ----------
    image : np.ndarray
        The image to find the connected components in.
    min_size : int
        The minimum size of the connected components.
    max_size : int
        The maximum size of the connected components.
    crop_components : bool, optional
        If True, the components will be cropped to the smallest possible size.
        The default is False.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is False.

    Returns
    -------
    tuple[list[np.ndarray], np.ndarray]
        A List of cropped masks with only one component and a big mask with all the components.
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
    # Check if the image is binary
    # check_binary_image(image, raise_exceptions=True)

    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        image, 8, cv2.CV_32S
    )

    # Create a smaller image for each component and a big image with all the components
    components = []
    total_components = np.zeros_like(image)
    for i in range(1, num_labels):
        # Check if the size of the component is within the min and max size
        if min_size < stats[i, cv2.CC_STAT_AREA] < max_size:
            # Create the full size mask, put the component in, crop away the
            # empty space and add it to the list
            component = np.zeros_like(image)
            component[labels == i] = 255

            # if crop_components:
            #     component = crop_mask(
            #         component, correct_rotation=False, show_steps=False
            #     )

            components.append(component)

            # Add the component to the total components
            total_components[labels == i] = 255

    if show_steps:
        # Show the original image, the original with the components marked and the total components
        images = {
            "Original image": image,
            "Image with components marked": total_components,
        }
        if 0 < len(components) < 10:
            for i, component in enumerate(components):
                images[f"Component {i}"] = component
        else:
            for i in range(10):
                images[f"Component {i}"] = components[i]
            print(f"Found {len(components)} components, not showing them all.")

        show_images(images)

    return components, total_components
