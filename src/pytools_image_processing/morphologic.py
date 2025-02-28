import cv2
import matplotlib.pyplot as plt
import numpy as np
from .utils import check_three_channel_image, show_images, check_grayscale_image
from .exceptions import ImageNotGrayscaleError, ImageNot3ChannelError
from typing import Tuple

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

    Change the saturation of an RGB image. The saturation is changed by changing the
    saturation channel of the image in the HSV color space.

    .. warning::
        The image is expected to be in RGB format, but any other 3 channel image will
        be accepted. This can cause unexpected results.

    Parameters
    ----------
    image : np.ndarray
        The image to change the saturation of, the image is expected to be in RGB format.
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
    ImageNot3ChannelError
        If the image is not a 3 channel image.
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

    check_three_channel_image(image, raise_exceptions=True)

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

    Change the brightness of an RGB image. The brightness is changed by changing the
    value channel of the image in the HSV color space.

    Parameters
    ----------
    image : np.ndarray
        The image to change the brightness of, the image is to be RGB format.
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
    ImageNot3ChannelError
        If the image is not a 3 channel image.
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

    check_three_channel_image(image, raise_exceptions=True)

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

    # Create the kernel and convolve the image with the kernel
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
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
        show_images(
            {
                "Original image": image,
                "Blurred image": image,
                "Difference between the two images": image - image,
            }
        )

    return image



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
