"""Image I/O and intensity transforms shared by tracking scripts."""

import numpy as np
from skimage.io import imread


def _max_projection(img_5d, time_stamp, channel, channel_axis_last):
    """
    Max-project a single channel across all z-slices for one timepoint.

    Parameters
    ----------
    img_5d : ndarray
        Source 5D image array.
    time_stamp : int
        Zero-based frame index.
    channel : int
        Channel index.
    channel_axis_last : bool
        True for `{T, Z, X, Y, C}`, False for `{T, Z, C, X, Y}`.
    """
    if channel_axis_last:
        return np.max(img_5d[time_stamp, :, :, :, channel], axis=0)
    return np.max(img_5d[time_stamp, :, channel, :, :], axis=0)


def read_projected_channels(
    img_path, time_stamp, spindle_channel, cell_channel, channel_axis_last
):
    """
    Read one frame from a 5D TIFF and return normalized spindle/cell channels.

    Parameters
    ----------
    img_path : str
        Path to the input TIFF.
    time_stamp : int
        Zero-based frame index.
    spindle_channel : int
        Channel index for spindle signal.
    cell_channel : int
        Channel index for cell or GFP signal.
    channel_axis_last : bool
        True for `{T, Z, X, Y, C}`, False for `{T, Z, C, X, Y}`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two float16 arrays in [0, 1] range:
        `(img_spindle_norm, img_cell_norm)`.
    """
    img = imread(img_path)

    img_spindle = _max_projection(img, time_stamp, spindle_channel, channel_axis_last)
    img_cell = _max_projection(img, time_stamp, cell_channel, channel_axis_last)

    img_spindle_norm = (img_spindle - img_spindle.min()) / (
        img_spindle.max() - img_spindle.min()
    )
    img_cell_norm = (img_cell - img_cell.min()) / (img_cell.max() - img_cell.min())

    return img_spindle_norm.astype(np.float16), img_cell_norm.astype(np.float16)


def auto_adjust(img_norm):
    """
    Apply ImageJ-style auto contrast to a normalized 2D image.

    This is a direct algorithmic rewrite used by the legacy scripts to improve
    visibility of dim spindle signal before segmentation.

    Parameters
    ----------
    img_norm : ndarray
        Input 2D image, typically normalized to [0, 1].

    Returns
    -------
    ndarray
        Contrast-adjusted image (float array).
    """
    im_min = np.min(img_norm)
    im_max = np.max(img_norm)

    hist_min = im_min
    hist_max = im_max
    histogram = np.histogram(img_norm, bins=256, range=(hist_min, hist_max))[0]
    bin_size = (hist_max - hist_min) / 256

    h, w = img_norm.shape
    pixel_count = h * w
    limit = pixel_count / 10
    const_auto_threshold = 5000
    auto_threshold = 0

    auto_threshold = (
        const_auto_threshold if auto_threshold <= 10 else auto_threshold / 2
    )
    threshold = int(pixel_count / auto_threshold)

    i = -1
    found = False
    while not found and i <= 255:
        i += 1
        count = histogram[i]
        if count > limit:
            count = 0
        found = count > threshold
    hmin = i
    found = False

    i = 256
    while not found and i > 0:
        i -= 1
        count = histogram[i]
        if count > limit:
            count = 0
        found = count > threshold
    hmax = i

    if hmax >= hmin:
        min_ = hist_min + hmin * bin_size
        max_ = hist_min + hmax * bin_size
        if min_ == max_:
            min_ = hist_min
            max_ = hist_max
    else:
        min_ = hist_min
        max_ = hist_max

    imr = (img_norm - min_) / (max_ - min_)

    return imr
