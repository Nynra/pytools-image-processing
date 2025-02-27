"""
This module contains some usefull errors that can be raised in the image processing module.
This way when catching errors it is easier to know what went wrong.
"""

class ImageNotGrayscaleError(Exception):
    """Raised when an image is not grayscale."""
    pass


class ImageNotBinaryError(Exception):
    """Raised when an image is not binary."""
    pass


class ImageNotRGBError(Exception):
    """Raised when an image is not RGB."""
    pass