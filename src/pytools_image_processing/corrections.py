import numpy as np
import copy


def correct_median_diff(self, inline=True):
    """
    Correct the image with the median difference
    """
    N = self.pixels
    # Difference of the pixel between two consecutive row
    N2 = N - np.vstack([N[:1, :], N[:-1, :]])
    # Take the median of the difference and cumsum them
    C = np.cumsum(np.median(N2, axis=1))
    # Extend the vector to a matrix (row copy)
    D = np.tile(C, (N.shape[0], 1)).T
    if inline:
        self.pixels = N - D
    else:
        new_img = copy.deepcopy(self)
        new_img.pixels = N - D
        return new_img


def correct_slope(self, inline=True):
    """
    Correct the image by subtracting a fitted slope along the y-axis
    """
    s = np.mean(self.pixels, axis=1)
    i = np.arange(s.shape[0])
    fit = np.polyfit(i, s, 1)
    if inline:
        self.pixels -= np.tile(
            np.polyval(fit, i).reshape(self.pixels.shape[0], 1),
            self.pixels.shape[1],
        )
        return self
    else:
        New = copy.deepcopy(self)
        New.pixels -= np.tile(
            np.polyval(fit, i).reshape(self.pixels.shape[0], 1),
            self.pixels.shape[1],
        )
        return New


def correct_plane(self, inline=True, mask=None):
    """
    Correct the image by subtracting a fitted 2D-plane on the data

    Parameters
    ----------
    inline : bool
        If True the data of the current image will be updated otherwise a new image is created
    mask : None or 2D numpy array
        If not None define on which pixels the data should be taken.
    """
    x = np.arange(self.pixels.shape[1])
    y = np.arange(self.pixels.shape[0])
    X0, Y0 = np.meshgrid(x, y)
    Z0 = self.pixels
    if mask is not None:
        X = X0[mask]
        Y = Y0[mask]
        Z = Z0[mask]
    else:
        X = X0
        Y = Y0
        Z = Z0
    A = np.column_stack((np.ones(Z.ravel().size), X.ravel(), Y.ravel()))
    c, resid, rank, sigma = np.linalg.lstsq(A, Z.ravel(), rcond=-1)
    if inline:
        self.pixels -= c[0] * np.ones(self.pixels.shape) + c[1] * X0 + c[2] * Y0
        return self
    else:
        New = copy.deepcopy(self)
        New.pixels -= c[0] * np.ones(self.pixels.shape) + c[1] * X0 + c[2] * Y0
        return New


def correct_lines(self, inline=True):
    """
    Subtract the average of each line for the image.

    if inline is True the current data are updated otherwise a new image with the corrected data is returned
    """
    if inline:
        self.pixels -= np.tile(
            np.mean(self.pixels, axis=1).T, (self.pixels.shape[1], 1)
        ).T
        return self
    else:
        New = copy.deepcopy(self)
        New.pixels -= np.tile(
            np.mean(self.pixels, axis=1).T, (self.pixels.shape[1], 1)
        ).T
        return New


def spline_offset(self, X, Y, Z=None, inline=True, ax=None, output="img", **kargs):
    """
    subtract a spline interpolated by points corrdinates.
    if Z is None, the image values will be used (default)
    """
    if ax is not None:
        if "num" in kargs and kargs["num"]:
            text_color = "k"
            if "text_color" in kargs:
                text_color = kargs["text_color"]
                del kargs["text_color"]
            for i in range(len(X)):
                l = self.pixels.shape[1] - X[i] < 20
                ax.annotate(
                    str(i),
                    (X[i], Y[i]),
                    ([5, -5][l], 0),
                    textcoords="offset pixels",
                    va="center",
                    ha=["left", "right"][l],
                    color=text_color,
                )
            del kargs["num"]
        ax.plot(X, Y, "o", **kargs)
    import scipy.interpolate

    T = np.flipud(self.pixels) - np.min(self.pixels)
    if Z is None:
        Z = [T[Y[i], X[i]] for i in range(len(X))]
    x = np.arange(self.pixels.shape[1])
    y = np.arange(self.pixels.shape[0])
    xx, yy = np.meshgrid(x, y)
    I = scipy.interpolate.SmoothBivariateSpline(X, Y, Z)
    z = I.ev(xx, yy)
    if inline:
        self.pixels -= z
        return z
    else:
        if output == "img":
            New = copy.deepcopy(self)
            New.pixels -= z
            return New
        elif output == "spline":
            return z
        else:
            raise ValueError("The output parameter should be either 'img' or 'spline'")


def zero_min(self, inline=True):
    """
    Shift the values so that the minimum becomes zero.
    """
    if inline:
        self.pixels -= np.min(self.pixels)
        return self
    else:
        N = copy.deepcopy(self)
        N.pixels -= np.min(N.pixels)
        return N


def get_radius_mask_from_center(self, radius):
    mask = np.zeros(self.pixels.shape, dtype=np.uint8)
    rr, cc = disk(
        center=(self.pixels.shape[0] // 2, self.pixels.shape[1] // 2), radius=radius
    )
    mask[rr, cc] = 1
    return mask
