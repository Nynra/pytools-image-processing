import cv2
import numpy as np
import os
from typing import List
from .utils import show_images, check_rgb_image, check_grayscale_image


def stack_keypoint_matching(
    images: List[np.ndarray], show_steps: bool = False
) -> np.ndarray:
    """Align and stack images by matching ORB keypoints

    This method is faster but less accurate than the perspective
    transformation method. The method expects all images to be of the
    same size and that they all exist.

    Parameters
    ----------
    images : List[np.ndarray]
        The images to stack
    show_steps : bool, optional
        Whether to show the steps, by default False

    Returns
    -------
    np.ndarray
        The stacked image
    """
    if not isinstance(images, list):
        raise TypeError("Images must be a list, not type {}".format(type(images)))
    if not all(isinstance(img, np.ndarray) for img in images):
        raise TypeError(
            "All images must be numpy arrays, not types {}".format(
                [type(img) for img in images]
            )
        )
    if not all(img.shape == images[0].shape for img in images):
        raise ValueError(
            "All images must have the same shape, not {}".format(
                [img.shape for img in images]
            )
        )
    if not isinstance(show_steps, bool):
        raise TypeError(
            "Show steps must be a bool, not type {}".format(type(show_steps))
        )

    orb = cv2.ORB_create()

    # disable OpenCL to because of bug in ORB in OpenCV 3.1
    cv2.ocl.setUseOpenCL(False)

    stacked_image = None
    first_image = None
    first_kp = None
    first_des = None
    for image in images:
        imageF = image.astype(np.float32) / 255

        # compute the descriptors with ORB
        kp = orb.detect(image, None)
        kp, des = orb.compute(image, kp)

        # create BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if first_image is None:
            # Save keypoints for first image
            stacked_image = imageF
            first_image = image
            first_kp = kp
            first_des = des
        else:
            # Find matches and sort them in the order of their distance
            matches = matcher.match(first_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32([first_kp[m.queryIdx].pt for m in matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate perspective transformation
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            dims = imageF.shape  # Might give errors
            w, h = dims[0], dims[1]
            imageF = cv2.warpPerspective(imageF, M, (h, w))
            stacked_image += imageF

    stacked_image /= len(images)
    stacked_image = (stacked_image * 255).astype(np.uint8)

    if show_steps:
        show_images(
            {
                "Original image": image,
                "Stacked image": stacked_image,
                "Difference between the two images": image - stacked_image,
            }
        )
    return stacked_image


def stack_ECC(images: List[np.ndarray], show_steps: bool = False) -> np.ndarray:
    """Align and stack images by estimating the perspective transformation

    This method is slower but more accurate than keypoint matching.
    The method expects all images to exist and be of the same size.

    Parameters
    ----------
    images : List[np.ndarray]
        The images to stack
    show_steps : bool, optional
        Whether to show the steps, by default False

    Returns
    -------
    np.ndarray
        The stacked image
    """
    if not isinstance(images, list):
        raise TypeError("Images must be a list, not type {}".format(type(images)))
    if not all(isinstance(img, np.ndarray) for img in images):
        raise TypeError(
            "All images must be numpy arrays, not types {}".format(
                [type(img) for img in images]
            )
        )
    if not all(img.shape == images[0].shape for img in images):
        raise ValueError(
            "All images must have the same shape, not {}".format(
                [img.shape for img in images]
            )
        )
    if not isinstance(show_steps, bool):
        raise TypeError(
            "Show steps must be a bool, not type {}".format(type(show_steps))
        )
    if check_grayscale_image(images[0], raise_exceptions=False):
        is_gray = True
    else:
        is_gray = False

    M = np.eye(3, 3, dtype=np.float32)

    first_image = None
    stacked_image = None

    for image in images:
        image = image.astype(np.float32) / 255
        if first_image is None:
            # convert to gray scale floating point image
            if is_gray:
                first_image = image
                stacked_image = image
            else:
                first_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                stacked_image = image
        else:
            # Estimate perspective transform
            s, M = cv2.findTransformECC(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if not is_gray else image,
                first_image,
                M,
                cv2.MOTION_HOMOGRAPHY,
            )
            dims = image.shape # Might give errors
            w, h = dims[0], dims[1]
            # Align image to first image
            image = cv2.warpPerspective(image, M, (h, w))
            stacked_image += image

    stacked_image /= len(images)
    stacked_image = (stacked_image * 255).astype(np.uint8)
    return stacked_image
