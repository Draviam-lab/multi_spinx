"""Segmentation helpers for spindle and GFP object extraction."""

import numpy as np
from scipy.ndimage import binary_fill_holes, label
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


def spindle_segmentation(img, lower_marker, higher_marker, padding):
    """
    Segment spindle-like objects with watershed + connected components.

    Parameters
    ----------
    img : ndarray
        Input 2D frame (normalized spindle channel).
    lower_marker : float
        Lower threshold marker for watershed background.
    higher_marker : float
        Upper threshold marker for watershed foreground.
    padding : int
        Extra pixels added around the square bounding box.

    Returns
    -------
    tuple
        `(seg_spindle, bbox_list, centroid_list, centroid_local_list)`.
        Bounding boxes are `(min_row, min_col, max_row, max_col)`.
    """
    markers = np.zeros_like(img)
    markers[img < lower_marker] = 1
    markers[img > higher_marker] = 2

    seg_spindle = watershed(img, markers)
    seg_spindle = binary_fill_holes(seg_spindle - 1)
    seg_spindle = remove_small_objects(seg_spindle, 900)

    spindle_instance, _ = label(seg_spindle)
    spindle_regions = regionprops(spindle_instance)

    bbox_list = []
    centroid_list = []
    centroid_local_list = []
    for i in range(0, len(spindle_regions)):
        minr, minc, maxr, maxc = spindle_regions[i].bbox

        center_row, center_col = (minr + maxr) / 2, (minc + maxc) / 2
        width, height = maxr - minr, maxc - minc
        size = max(width, height) + 2 * padding

        minr, maxr = center_row - size / 2, center_row + size / 2
        minc, maxc = center_col - size / 2, center_col + size / 2

        minr, minc = max(0, minr), max(0, minc)
        maxr, maxc = min(img.shape[0], maxr), min(img.shape[1], maxc)

        if minr > 0 and minc > 0 and maxr < img.shape[0] and maxc < img.shape[1]:
            bbox_list.append((minr, minc, maxr, maxc))
            centroid_list.append(spindle_regions[i].centroid)
            centroid_local_list.append(spindle_regions[i].centroid_local)

    return seg_spindle, bbox_list, centroid_list, centroid_local_list


def gfp_segmentation(img, lower_marker, higher_marker, small_area, large_area):
    """
    Segment GFP puncta-like objects with watershed and area filtering.

    Parameters
    ----------
    img : ndarray
        Input 2D frame (masked/normalized GFP channel).
    lower_marker : float
        Lower threshold marker for watershed background.
    higher_marker : float
        Upper threshold marker for watershed foreground.
    small_area : int
        Minimum connected-component area to keep.
    large_area : int
        Upper area bound used in XOR filtering logic.

    Returns
    -------
    tuple
        `(seg_gfp, bbox_list, centroid_list, centroid_local_list)`.
    """
    markers = np.zeros_like(img)
    markers[img < lower_marker] = 1
    markers[img > higher_marker] = 2

    seg_gfp = watershed(img, markers)
    seg_gfp = binary_fill_holes(seg_gfp - 1)

    small_removed = remove_small_objects(seg_gfp, small_area)
    mid_removed = remove_small_objects(seg_gfp, large_area)
    seg_gfp = small_removed ^ mid_removed

    gfp_instance, _ = label(seg_gfp)
    gfp_regions = regionprops(gfp_instance)

    bbox_list = []
    centroid_list = []
    centroid_local_list = []
    for i in range(0, len(gfp_regions)):
        minr, minc, maxr, maxc = gfp_regions[i].bbox
        bbox_list.append((minr, minc, maxr, maxc))
        centroid_list.append(gfp_regions[i].centroid)
        centroid_local_list.append(gfp_regions[i].centroid_local)

    return seg_gfp, bbox_list, centroid_list, centroid_local_list
