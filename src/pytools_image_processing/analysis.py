import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from .utils import (
    check_three_channel_image,
    show_images,
    check_binary_image,
    check_grayscale_image,
)
from .exceptions import ImageNotBinaryError, ImageNotGrayscaleError, ImageNot3ChannelError
import copy
from pytools_image_processing.morphologic import remap_image_intensity, normalize_image
from pytools_image_processing.utils import show_images


def plot_intensity_profile(image: np.ndarray, show_steps: bool=False) -> Tuple[np.ndarray, np.ndarray]:
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
        raise TypeError(
            "Image should be a numpy array not type {}".format(type(image))
        )
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



def get_bounding_rect(
    contour: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int, float]]:
    """Get the bounding rectangle of a contour.

    Get the bounding rectangle of a contour and the coordinates of the rectangle.
    The contour is a mask of the object in the image.

    Parameters
    ----------
    contour : np.ndarray
        The contour to get the bounding rectangle of.

    Returns
    -------
    tuple[np.ndarray, tuple[int, int, int, int, float]]
        The bounding rectangle and the coordinates of the rectangle.
        The coordinates are (x, y, w, h, a) where a is the angle of
        the rectangle.
    """
    if not isinstance(contour, np.ndarray):
        raise ValueError(
            "contour should be a numpy array not type {}".format(type(contour))
        )
    # As we expect a mask check for a binary image
    check_binary_image(contour, raise_exceptions=True, enforce_boolean=False)

    # Get the minimum area rectangle of the mask
    points = np.argwhere(contour.transpose(1, 0))
    rect = cv2.minAreaRect(points)
    (x, y), (w, h), a = rect  # a - angle
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    return box, (int(x), int(y), int(w), int(h), float(a - 90))


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

    Raises
    ------
    ImageNotGrayscaleError
        If the image is not a grayscale image.
    """
    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(image, np.ndarray):
        raise TypeError(
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
    show_steps: bool = False,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Finds the connected components in the binary image.

    This function uses the :func:`cv2.connectedComponentsWithStats` function to find
    the connected components in a binary image. It then creates a smaller
    image for each component and a big image with all the components.

    Parameters
    ----------
    image : np.ndarray
        The image to find the connected components in.
    min_size : int
        The minimum size of the connected components.
    max_size : int
        The maximum size of the connected components.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is False.

    Returns
    -------
    tuple[list[np.ndarray], np.ndarray]
        A List of cropped masks with only one component and a big mask with all the components.

    Raises
    ------
    ImageNotBinaryError
        If the image is not a binary image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Image should be a numpy array not type {}".format(type(image))
        )
    if not isinstance(min_size, int):
        raise TypeError(
            "min_size should be an integer not type {}".format(type(min_size))
        )
    if not isinstance(max_size, int):
        raise TypeError(
            "max_size should be an integer not type {}".format(type(max_size))
        )
    if not isinstance(show_steps, bool):
        raise TypeError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    # Check if the image is binary
    check_binary_image(image, raise_exceptions=True, enforce_boolean=False)

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
        elif len(components) >= 10:
            for i in range(10):
                images[f"Component {i}"] = components[i]
            print(f"Found {len(components)} components, not showing them all.")
        else:
            print("No components found.")

        show_images(images)

    return components, total_components


def separate_objects(
    image: np.ndarray,
    normalize: bool = True,
    mask_range: tuple[float] = (0.05, 0.8),
    invert_image: bool = False,
    dilate_iterations: int = 30,
    show_steps: bool = False,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Separate objects in a SEM image.

    This function takes a grayscale image and returns a list of images
    with the separated objects. The function finds the object by using a
    Gaussian blur and a OTSU threshold. During OTSU thresholding a lot
    of noise will be found so these are removed by defining a minimum
    and maximum size of the components. The components are then dilated
    to make sure the whole object is included in the cropped image.

    .. attention::

        This function assumes that the line profile is higher than the
        substrate around the lines. This means that when only development 
        is done the reverse will be true and the lines will be lower than
        the substrate. This will cause the function to either fail or
        see the space between the lines as the line (in the case of a pre
        cropped image)

    Parameters
    ----------
    image : np.ndarray
        The grayscale image.
    normalize : bool, optional
        If True, normalize the intensity of the image. The default is True.
    correct_rotation : bool, optional
        If True, correct the rotation of the image. The default is True.
    mask_range : tuple[float], optional
        The range of the mask size in percentage of the image size.
        The default is (0.05, 0.5) meaning the mask should be between
        5% and 50% of the image size.
    invert_image : bool, optional
        If True, invert the image. The default is False. This is useful
        when the lines are lower than the substrate.
    dilate_iterations : int, optional
        The number of iterations to dilate the mask. The default is 30.
        The mask is dilated to make sure the whole object is included in the
        cropped image.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is False.

    Returns
    -------
    tuple[list[np.ndarray], list[float]]
        The separated object masks and the angles of the objects.
        The masks are the components found in the image.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            "image should be a numpy array not type {}".format(type(image))
        )
    if not len(image.shape) == 2:
        raise ValueError(
            "image should be a grayscale image not shape {}".format(image.shape)
        )
    if not isinstance(normalize, bool):
        raise ValueError(
            "normalize should be a boolean not type {}".format(type(normalize))
        )
    if not isinstance(mask_range, (tuple, list)):
        raise ValueError(
            "mask_range should be a tuple not type {}".format(type(mask_range))
        )
    if not len(mask_range) == 2:
        raise ValueError(
            "mask_range should be a tuple of length 2 not {}".format(len(mask_range))
        )
    if not isinstance(mask_range[0], (int, float)) or not isinstance(
        mask_range[1], (int, float)
    ):
        raise ValueError(
            "mask_range should be a tuple of integers or floats not {}, {}".format(
                type(mask_range[0]), type(mask_range[1])
            )
        )
    if mask_range[0] < 0 or mask_range[0] > mask_range[1]:
        raise ValueError(
            "mask_range should be a tuple of two positive numbers "
            "where the first is smaller than the second, so not {} and {}".format(
                mask_range[0], mask_range[1]
            )
        )
    if not isinstance(dilate_iterations, int):
        raise ValueError(
            "dilate_iterations should be an integer not type {}".format(
                type(dilate_iterations)
            )
        )
    if not isinstance(show_steps, bool):
        raise ValueError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )

    # Normalize the image to increase the contrast
    if normalize:
        norm_image = normalize_image(image)
    else:
        norm_image = image

    # Remove some noise and use a threshold to get the line positions
    norm_image = cv2.GaussianBlur(norm_image, (3, 3), 0)
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # To find the contours we need to define a minimum and maximum size
    # otherwise every pixel cluster will be considered a component
    surface = np.prod(image.shape)
    min_size = int(surface * mask_range[0])
    max_size = int(surface * mask_range[1])

    # Find components
    components, component_img = find_components(
        image=mask,
        min_size=min_size,
        max_size=max_size,
        show_steps=False,
        # crop_components=False,
    )
    if len(components) == 0:
        raise ValueError("No components found in the mask!")

    # Now we have the components we can use them to separate the objects
    # in the original image
    angles = []  # Store the angles of the objects
    for component in components:
        # Dilate the mask a bit to make sure we get the whole object
        dil_component = cv2.dilate(
            component, np.ones((3, 3), np.uint8), iterations=dilate_iterations
        )
        coords = get_bounding_rect(dil_component)
        angles.append(coords[-1])

    if show_steps:
        images = {
            "Component {}, a={}".format(i, angles[i]): component
            for i, component in enumerate(components)
        }
        show_images(images)

    return components, angles



def mark_objects(
    image: np.ndarray, masks: list[np.ndarray], start_id: int, show_steps: bool = False
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Mark the objects to keep track of them.

    During batch processing it becomes harder to keep track of the
    different objects. This function can be used to mark the different
    objects in the image. The objects will me marked with a bounding box
    and a number, starting from 1 in the order of the given masks.

    Parameters
    ----------
    image : np.ndarray
        The image to mark.
    masks : list[np.ndarray]
        The masks of the objects.
    start_id : int
        The number to start counting from.
    show_steps : bool, optional
        If True, show the steps of the calculation. The default is False.

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray]]
        The marked image and list of ID's.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            "image should be a numpy array not type {}".format(type(image))
        )
    if not len(image.shape) == 2:
        raise ValueError(
            "image should be a grayscale image not shape {}".format(image.shape)
        )
    if not isinstance(masks, list):
        raise ValueError("masks should be a list not type {}".format(type(masks)))
    if not all(isinstance(mask, np.ndarray) for mask in masks):
        raise ValueError(
            "masks should be a list of numpy arrays not {}".format(
                [type(mask) for mask in masks]
            )
        )
    if not all(mask.shape == image.shape for mask in masks):
        raise ValueError(
            "masks should have the same shape as the image not {}".format(
                [mask.shape for mask in masks]
            )
        )
    if not isinstance(start_id, int):
        raise ValueError(
            "start_id should be an integer not type {}".format(type(start_id))
        )
    if not isinstance(show_steps, bool):
        raise ValueError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )

    # Create a copy of the image
    marked_image = image.copy()

    # Draw the masks on the image
    ids = []
    for i, mask in enumerate(masks):
        # Get the bounding box of the mask
        box, (x, y, w, h, angle) = get_bounding_rect(mask)

        # Draw the bounding box using the point box so it is rotated
        cv2.polylines(marked_image, [box], isClosed=True, color=255, thickness=2)

        # Draw the number of the object
        id = str(start_id + i + 1)
        cv2.putText(
            marked_image,
            id,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            255,
            thickness=2,
        )
        ids.append(id)


    if show_steps:
        images = {"Marked image": marked_image}
        show_images(images)

    return marked_image, ids


def get_object(
    image: np.ndarray, mask: np.ndarray, dil_iter: int = 10, show_steps: bool = False
) -> np.ndarray:
    """Get the object from the image using the mask and angle.

    Parameters
    ----------
    image : np.ndarray
        The image with the object.
    mask : np.ndarray
        The mask of the object.
    dil_iter: int, optional
        The number of iterations to dilate the mask. The default is 10.
    show_steps : bool, optional
        If True, show the steps of the calculation. The default is False.

    Returns
    -------
    np.ndarray
        The object from the image.
    """
    # Use the bounding box of the mask to get the center of rotation
    _, (x, y, w, h, angle) = get_bounding_rect(mask)
    center_point = (x + w // 2, y + h // 2)

    # Rotate the image and the mask
    rot_matrix = cv2.getRotationMatrix2D(center=center_point, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
    rotated_mask = cv2.warpAffine(mask, rot_matrix, (image.shape[1], image.shape[0]))

    # Dilate the mask to make sure the whole object is included
    dilated_mask = cv2.dilate(
        rotated_mask, np.ones((3, 3), np.uint8), iterations=dil_iter
    )

    # Get the bounding box of the dilated mask
    rect = cv2.boundingRect(dilated_mask)
    x, y, w, h = rect

    # Crop the image
    cropped_image = rotated_image[y : y + h, x : x + w]

    # Rotation causes some annoing black pixels around the object in
    # some cases so we need to remove them. The used method is not
    # perfect but it works for now.
    # Create a mask that only detects black pixels
    mask = cv2.inRange(cropped_image, 0, 0)
    # Find the contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Each of the objects should touch one of the image borders
    # If not the object is not the object we are looking for
    img_x, img_y = cropped_image.shape
    if len(contours) != 0:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x == 0:
                # Touches the left border so crop from the left
                cropped_image = cropped_image[:, w:]
            elif y == 0:
                # Touches the top border so crop from the top
                cropped_image = cropped_image[h:, :]

            if x + w == img_x:
                # Touches the right border so crop from the right
                cropped_image = cropped_image[:, :w]
            elif y + h == img_y:
                # Touches the bottom border so crop from the bottom
                cropped_image = cropped_image[:h, :]
            else:
                continue

    if show_steps:
        images = {
            "Original image": image,
            "Rotated mask": rotated_mask,
            "Dilated mask": dilated_mask,
            "Cropped image": cropped_image,
        }
        show_images(images)

    return cropped_image

