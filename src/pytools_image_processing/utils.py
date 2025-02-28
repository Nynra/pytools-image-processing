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
    box = np.int0(box)  # turn into ints
    return box, (int(x), int(y), int(w), int(h), float(a - 90))


def rotate_image(
    image: np.ndarray,
    angle: float,
    centerpoint: tuple[int, int] = None,
    crop_image: bool = True,
    show_steps: bool = False,
) -> np.ndarray:
    """Rotates the given image about the given center point.

    Rotate the image and get rid of the black space due to the rotation.

    .. attention::

        Because this function will get rid of the black space due rotation, the
        image will be cropped, so the output image will be a different
        size from the original.

    Parameters
    ----------
    image : np.ndarray
        The image to rotate.
    angle : float
        The angle to rotate the image by.
    centerpoint : tuple[int,int], optional
        The center point to rotate the image about. The default is None. If
        no centerpoint is given, the image will be rotated about its center.
    crop_image : bool, optional
        If True after rotation the image will be cropped to the biggest possible
        size without black space. The default is True.
    show_steps : bool, optional
        If True, show the steps of the rotation. The default is False.

    Returns
    -------
    np.ndarray
        The rotated image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array, not type {}.".format(type(image)))
    if not isinstance(angle, (int, float)):
        raise TypeError(
            "Angle must be an integer or float, not type {}.".format(type(angle))
        )
    if centerpoint is not None:
        if not isinstance(centerpoint, (tuple, list)):
            raise TypeError(
                "Centerpoint must be a tuple, not type {}.".format(type(centerpoint))
            )
        if not all(isinstance(i, int) for i in centerpoint):
            raise TypeError(
                "Centerpoint must contain two integers, not types {} and {}.".format(
                    type(centerpoint[0]), type(centerpoint[1])
                )
            )
    if not isinstance(crop_image, bool):
        raise TypeError(
            "Crop_image must be a boolean, not type {}.".format(type(crop_image))
        )
    if not isinstance(show_steps, bool):
        raise TypeError(
            "Show_steps must be a boolean, not type {}.".format(type(show_steps))
        )
    
    # Check if we have to do anything
    if abs(angle) in (0, 180, 360):
        # Nothing to do here
        return image

    image_size = (image.shape[1], image.shape[0])

    # If a centerpoint is given use it otherwise use image center
    if centerpoint is not None:
        image_center = centerpoint
    else:
        image_center = tuple(np.array(image_size) / 2)

    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Rotate the image and the mask
    rotated_image = cv2.warpAffine(
        image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR
    )

    if crop_image:
        if len(image.shape) != 2:
            # Convert the image to grayscale
            rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to detect black areas
        _, thresh = cv2.threshold(rotated_image, 1, 255, cv2.THRESH_BINARY)

        # Find the contours of the non black areas
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Get the largest contour
        x, y, w, h = cv2.boundingRect(contours[0])

        # Crop the image
        rotated_image = rotated_image[y : y + h, x : x + w]

    if show_steps:
        images = {
            "original": image,
            "rotated": rotated_image,
        }
        show_images(images)

    return rotated_image


def rotated_rect_with_max_area(
    width: float, height: float, angle: float
) -> Tuple[float, float]:
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    Converted to Python by Aaron Snoswell

    Source: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders

    .. attention::
        This method is mostly used for cropping images that have been rotated. With
        a big enough angle the image will be cropped to a very small size.

    Parameters
    ----------
    width : float
        The width of the rectangle.
    height : float
        The height of the rectangle.
    angle : float
        The angle of rotation.

    Returns
    -------
    Tuple[float, float]
        The width and height of the largest possible axis-aligned rectangle.
    """
    if width <= 0 or height <= 0:
        return 0, 0

    width_is_longer = width >= height
    side_long, side_short = (width, height) if width_is_longer else (height, width)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (width * cos_a - height * sin_a) / cos_2a, (
            height * cos_a - width * sin_a
        ) / cos_2a

    return wr, hr


# def crop_mask(
#     mask: np.ndarray, correct_rotation: bool = False, show_steps: bool = True
# ) -> tuple[np.ndarray, float]:
#     """Crops the mask to the smallest possible size.

#     Crop the mask to the smallest possible size. The mask is used to find the bounding box
#     of the object in the image.

#     Parameters
#     ----------
#     mask : np.ndarray
#         The mask to crop.
#     correct_rotation : bool, optional
#         If True, correct the rotation of the mask. The default is False.
#     show_steps : bool, optional
#         If True, show the steps of the conversion. The default is True.

#     Returns
#     -------
#     tuple[np.ndarray, float]
#         The cropped mask and the angle of rotation.
#     """
#     if not isinstance(mask, np.ndarray):
#         raise ValueError("mask should be a numpy array not type {}".format(type(mask)))
#     if not isinstance(correct_rotation, bool):
#         raise ValueError(
#             "correct_rotation should be a boolean not type {}".format(
#                 type(correct_rotation)
#             )
#         )
#     if not isinstance(show_steps, bool):
#         raise ValueError(
#             "show_steps should be a boolean not type {}".format(type(show_steps))
#         )

#     # Get the bounding rectangle of the mask
#     _, (x, y, w, h, a) = get_bounding_rect(mask)

#     if correct_rotation:
#         # Correct the rotation of the mask using a and opencv
#         mask_center = (x + w // 2, y + h // 2)
#         rot_mat = cv2.getRotationMatrix2D(mask_center, a, 1.0)
#         rotated_mask = cv2.warpAffine(
#             mask, rot_mat, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR
#         )

#         # Get the bounding rectangle of the rotated mask
#         _, (x, y, w, h, _) = get_bounding_rect(rotated_mask)
#     else:
#         rotated_mask = mask

#     # Crop the mask
#     cropped_mask = rotated_mask[y : y + h, x : x + w]

#     return cropped_mask, a


def crop_image(
    image: np.ndarray,
    mask: np.ndarray,
    correct_rotation: bool = False,
    show_steps: bool = False,
) -> tuple[np.ndarray, float]:
    """Crop the image to the smallest possible size.

    Apply the mask to the image and crop the image to the smallest possible size.
    The mask is used to find the bounding box of the object in the image.

    .. warning::

        The mask should only contain the object that should be cropped. If
        the mask contains multiple objects, the result will be incorrect.

    Parameters
    ----------
    image : np.ndarray
        The image to crop.
    mask : np.ndarray
        The mask to apply to the image. The mask contains the object that should be cropped.
    correct_rotation : bool, optional
        If True, correct the rotation of the image. The default is False.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is False.

    Returns
    -------
    tuple[np.ndarray, float]
        The cropped image and the angle of rotation.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            "image should be a numpy array not type {}".format(type(image))
        )
    if not isinstance(mask, np.ndarray):
        raise ValueError("mask should be a numpy array not type {}".format(type(mask)))
    if not isinstance(correct_rotation, bool):
        raise ValueError(
            "correct_rotation should be a boolean not type {}".format(
                type(correct_rotation)
            )
        )
    if not isinstance(show_steps, bool):
        raise ValueError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )

    # Apply the mask to the image and get the boundingbox
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    box, (x, y, w, h, a) = get_bounding_rect(mask)

    if correct_rotation:
        # Create the rotation matrix
        mask_center = (x + w // 2, y + h // 2)
        rot_mat = cv2.getRotationMatrix2D(mask_center, a, 1.0)

        # Rotate the image and the mask
        rotated_image = cv2.warpAffine(
            image,
            rot_mat,
            (masked_image.shape[1], masked_image.shape[0]),
            flags=cv2.INTER_LINEAR,
        )
        rotated_mask = cv2.warpAffine(
            mask,
            rot_mat,
            (masked_image.shape[1], masked_image.shape[0]),
            flags=cv2.INTER_LINEAR,
        )

        # Get the bounding rectangle of the rotated image
        box, _ = get_bounding_rect(rotated_mask)

        # img = cv2.drawContours(rotated_image, [box], 0, (0, 255, 0), 2)
        # show_images({"Rotated image": img})

    else:
        rotated_image = masked_image
        rotated_mask = mask

    # Crop the image using the box
    cropped_image = rotated_image[box[1][1] : box[0][1], box[1][0] : box[2][0]]

    if show_steps:
        show_images(
            {
                "Original image": image,
                "Masked image": masked_image,
                "Rotated image": rotated_image,
                "Cropped image": cropped_image,
            }
        )

    return cropped_image, a
