# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:24:57 2023

@author: binghao chai

This script tracks the movements of multi spindles in a microscopy biology movie.
Generally, the script outputs detected spindles and their underlying brightfield 
cells as cropped images to fit the existing SpinX modules.

The cropped spindle images should be sent to SpinX-spindle module for spindle 
segmentation, and the cropped brightfield cell cortex images should be sent to 
the SpinX-cell-cortex module for cell cortex segmentation, and finally the
outputs of SpinX-spindle and SpinX-cell-cortex should be sent to SpinX-modelling
module for 3D modelling.

Parameters
----------
input_img: str
    The input source image for nucleus counting (multi-stack tiff).

time_stamp: int
    Define the start frame to track spindles, frame ID starting from 1, default 
    set to 1.

z_slice: int
    The selected z-slice reference, starting from 1.

spindle_channel: int
    The spindle channel ID, starting from 0.

cell_channel: int
    The cell (or brightfield) channel ID, starting from 0.

padding: int
    Define how many pixels to extend for each side of the bounding boxes to make 
    them larger, default value set to 0.

output: str
    Define the output folder path.

nr_frames: int
    Define how many frames to track the movie.
    
Returns
-------
# TODO: TBC

"""

# package import
import time
import argparse
import warnings
import os

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import Counter

from skimage import io
from skimage import transform

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore") # ignore warnings
since = time.time()

# arguments definition
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input_img",
        type = str, 
        default = "F:/Dropbox/Postdoc_QMUL/workspace/multispindle/data/exp2022_H1299_pi-EB1-GFP_EB3-mKate2_SiR-DNA_set21_DMSO-1-5_CilioDi-5uM-6-10_1_01_R3D.tif", 
        help = "the input source image for nucleus counting (multi-stack tiff)" 
        )
    parser.add_argument(
        # the time-stamp starts from 0, 
        "--time_stamp",
        type = int, 
        default = 0, 
        help = "define the start frame to track spindles, frame ID starting from 0, default set to 0" 
        )
    parser.add_argument(
        # the z-slice starts from 1, 
        # so when coding, need to use (opt.z_slice-1) as the time-stamp reference
        "--z_slice",
        type = int, 
        default = 3, 
        help = "the selected z-slice reference, starting from 1" 
        )
    parser.add_argument(
        # the spindle channel ID starts from 0
        "--spindle_channel",
        type = int, 
        default = 0, 
        help = "the spindle channel ID, starting from 0" 
        )
    parser.add_argument(
        # the brightfield/cell channel ID starts from 0
        "--cell_channel",
        type = int, 
        default = 3, 
        help = "the cell (or brightfield) channel ID, starting from 0" 
        )
    parser.add_argument(
        # the bounding box padding
        "--padding",
        type = int, 
        default = 40, 
        help = "how many pixels to extend for each side of the bounding boxes \
        to make them larger, default value set to 0" 
        )
    parser.add_argument(
        "--output",
        type = str, 
        default = "F:/Dropbox/Postdoc_QMUL/workspace/multispindle/output", 
        help = "define the output folder path" 
        )
    parser.add_argument(
        "--nr_frames",
        type = int, 
        default = 6, 
        help = "define how many frames to track the movie" 
        )

    opt = parser.parse_args()
    print(opt)

def img_read(img_path, time_stamp, z_slice, spindle_channel, cell_channel):
    """
    This function read the specific image channels for specific time frame under
    specific z-slide. 
    
    Parameters
    ----------
    img_path: str
        The input source image for nucleus counting (multi-stack tiff).
        
    time_stamp: int
        Define the start frame to track spindles, frame ID starting from 1, 
        default set to 1.
        
    z_slice: int
        The selected z-slice reference, starting from 1.
        
    spindle_channel: int
        The spindle channel ID, starting from 0.
        
    cell_channel: int
        The cell cortex (brightfield) channel ID, starting from 0.

    Returns
    -------
    img_spindle_norm: ndarray (2D)
        Data array stands for the the normalised (0-1 scale) spindle image.
    
    img_cell_norm: ndarray (2D)
        Data array stands for the the normalised (0-1 scale) cell cortex image.
        
    """

    from skimage.io import imread

    # source image read, the sample image (stacked-tiff) is at size {TT, ZZ, XX, YY, CC}
    # TT: time-stamp, ZZ: z-slice, XX and YY: size, CC: channels
    img = imread(img_path)
    img_spindle = img[time_stamp, z_slice - 1, :, :, spindle_channel]
    img_cell= img[time_stamp, z_slice - 1, :, :, cell_channel]
    # normalisation to the [0, 1] scale for img_spindle and img_cell
    img_spindle_norm = (img_spindle - img_spindle.min())/(img_spindle.max() - img_spindle.min())
    img_cell_norm = (img_cell - img_cell.min())/(img_cell.max() - img_cell.min())
    
    return img_spindle_norm, img_cell_norm

def spindle_segmentation(img):
    """
    This function segments the spindles using watershed method. The input of this
    function is a still image (in array of float64), and the outputs are the 
    segmented spindles (a bool/binary mask), the bounding box of each spindle
    (detected objects), the centroid coordinators and local centroid (relating 
    to the bounding box) of each spindle.
    
    Parameters
    ----------
    img: ndarray (2D)
        Data array stands for the the normalised (0-1 scale) spindle image.
        
    Returns
    -------
    seg_spindle: ndarray of bool (2D)
        The binary segmentation map.
        
    bbox_list: list
        The list of bounding boxes (min_row, min_col, max_row, max_col) for 
        each detected spindle.
        
    centroid_list: list
        The list of centroid (row, col) for each detected spindle.
    
    centroid_local_list: list
        The list of local centroid (row, col) relating to bounding box for 
        each detected spindle.
        
    """
    
    from scipy.ndimage import binary_fill_holes, label
    from skimage.segmentation import watershed
    from skimage.morphology import remove_small_objects
    from skimage.measure import regionprops
    
    # segmentation of the spindle(s) using the traditional watershed method
    # find the watershed markers of the background and the nuclei
    markers = np.zeros_like(img)
    markers[img < 0.3] = 1
    markers[img > 0.4] = 2
    # watershed segmentation of the spindles
    seg_spindle = watershed(img, markers)
    seg_spindle = binary_fill_holes(seg_spindle- 1)
    # remove small objects with boolean input "seg"
    seg_spindle = remove_small_objects(seg_spindle, 900)
        
    # generate spindle instance map based on the conventional watershed segmentation
    spindle_instance, nr_spindle = label(seg_spindle)
    # spindle regions cropping using skimage.measure.regionprops
    # refer to https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
    spindle_regions = regionprops(spindle_instance)
    
    # traversal the properties of each spindle
    bbox_list = []
    centroid_list= []
    centroid_local_list = []
    for i in range(0, len(spindle_regions)):
        
        # bounding box (min_row, min_col, max_row, max_col)
        # pixels belonging to the bounding box are in the half-open interval 
        # [min_row; max_row) and [min_col; max_col)
        
        # make the bounding box a square rather than rectangle
        minr, minc, maxr, maxc = spindle_regions[i].bbox # load original bounding box
        # Compute the center, width, and height of the bounding box
        center_row, center_col = (minr + maxr) / 2, (minc + maxc) / 2
        width, height = maxr - minr, maxc - minc
        # Compute the size of the square bounding box by taking the maximum of width and height
        size = max(width, height) + 2 * opt.padding  # add padding to both sides of the bounding box
        # Compute the new bounding box coordinates
        minr, maxr = center_row - size / 2, center_row + size / 2
        minc, maxc = center_col - size / 2, center_col + size / 2
        # Ensure the bounding box does not go beyond the image boundaries
        minr, minc = max(0, minr), max(0, minc)
        maxr, maxc = min(img.shape[0], maxr), min(img.shape[1], maxc)  
        
        # append the new bounding box to the list, 
        # only append the new bounding box if it does not touch the image boundary
        if minr > 0 and minc > 0 and maxr < img.shape[0] and maxc < img.shape[1]:
            bbox_list.append((minr, minc, maxr, maxc))
        
            # centroidarray coordinate tuple (row, col)
            centroid_list.append(spindle_regions[i].centroid)
            # centroid_local shows the centroid coordinate tuple (row, col), 
            # which is relative to region bounding box
            centroid_local_list.append(spindle_regions[i].centroid_local)
    
    # define the function returns
    return seg_spindle, bbox_list, centroid_list, centroid_local_list

# draw bounding box on top of the original images for illustration purpose
def bounding_box_plot(img, bbox_list):
    """
    This function plots the bounding box for illustration purpose.

    Parameters
    ----------
    img: ndarray (2D)
        Data array stands for the the normalised (0-1 scale) spindle image.
        
    bbox_list: list
        The list of bounding boxes (min_row, min_col, max_row, max_col) for 
        each detected spindle.

    Returns
    -------
    Currently none, the function only makes the plot.
    
    """
    
    import matplotlib.patches as mpatches
    
    # TODO: this function could be extended for generating the overlapping image
    # or movies (i.e., draw bounding boxes on top of the spindle and cell images 
    # or movies) as one of the script outputs.
    
    # define the figure and plot the original image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    
    # draw bounding boxes accordingly on the original image
    for bboxes in bbox_list:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = bboxes
        rect = mpatches.Rectangle(
            (minc, minr), maxc - minc, maxr - minr, 
            fill = False, edgecolor = 'red', linewidth = 4)
        ax.add_patch(rect)
    
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()  
 
# # Note: the below lines are for functions img_read, spindle_segmentation 
# # and bounding_box_plot testing
# # TODO: should be removed once script is completed
# img_spindle_norm, img_cell_norm = img_read(
#     f"{opt.input_img}", 
#     opt.time_stamp, 
#     opt.z_slice, 
#     opt.spindle_channel, 
#     opt.cell_channel)
# seg_spindle, bbox_list, centroid_list, centroid_local_list = spindle_segmentation(img_spindle_norm)
# bounding_box_plot(img_spindle_norm, bbox_list)
# bounding_box_plot(img_cell_norm, bbox_list)

# # TODO: these to be put within the tracking code - these to be put as a function
# # export cropped images (spindle channel and cell channel) based on bounding boxes
# # create output directories for cropped images (spindles & cells respectively) 
# # if they doesn't exist
# os.makedirs(f"{opt.output}/spindle", exist_ok = True)
# os.makedirs(f"{opt.output}/cell", exist_ok = True)

# for i, bbox in enumerate(bbox_list):
#     minr, minc, maxr, maxc = map(int, bbox)
#     # image operation on the normalised channels
#     cropped_spindle = img_spindle_norm[minr:maxr, minc:maxc]
#     cropped_cell = img_cell_norm[minr:maxr, minc:maxc]
    
#     # Resize the cropped images to the desired output size
#     output_size = (450, 450) # Define the size of the output images
#     resized_spindle = transform.resize(cropped_spindle, output_size)
#     resized_cell = transform.resize(cropped_cell, output_size)
    
#     # Save the cropped images as single-channel TIFF files
#     io.imsave(os.path.join(f"{opt.output}/spindle", f"spindle_{i}.tif"), resized_spindle)
#     io.imsave(os.path.join(f"{opt.output}/cell", f"cell_{i}.tif"), resized_cell)
    
# list to store the tracked spindles across all frames,
# with an additional tracked_spindle_number field indicating the identity of 
# the spindle across frames.
tracked_spindles = []

# process each frame
for frame_number in range(opt.time_stamp, opt.time_stamp + opt.nr_frames):
    # frame_number here is not the absolute frame_number of the multi-stacked tiff
    # but the relative frame_number in the [start_time_stamp - 1, end_time_stamp) range.
    
    # image read for the current frame,
    # the spindle and cell cortex channels are both normalised
    img_spindle_norm, img_cell_norm = img_read(
        f"{opt.input_img}", 
        frame_number, 
        opt.z_slice, 
        opt.spindle_channel, 
        opt.cell_channel)
    # perform spindle segmentation and bounding box generation for the current frame
    seg_spindle, bbox_list, centroid_list, _ = spindle_segmentation(img_spindle_norm)
    
    # list to store the spindles in the current frame
    current_frame_spindles = []

    # traverse the properties of each spindle
    for i in range(len(bbox_list)):
        # extract the bounding box and centroid indormation of the spindles
        minr = bbox_list[i][0]
        minc = bbox_list[i][1]
        maxr = bbox_list[i][2]
        maxc = bbox_list[i][3]
        centroid_row = centroid_list[i][0]
        centroid_col = centroid_list[i][1]

        # compute the area of the bounding box
        area = (maxr - minr) * (maxc - minc)

        # Store the spindle in the current frame list
        current_frame_spindles.append({
            'frame_number': frame_number,
            'spindle_number': i,
            'bounding_box': (minr, minc, maxr, maxc),
            'centroid': (centroid_row, centroid_col),
            'area': area,
            'tracked_spindle_number': None,  # initialize tracked_spindle_number
        })

    # if this is the first frame, just store the spindles without tracking
    # if frame_number == opt.time_stamp:
    #     tracked_spindles.extend(current_frame_spindles)
    if frame_number == opt.time_stamp:
        for i, spindle in enumerate(current_frame_spindles):
            spindle['tracked_spindle_number'] = i
        tracked_spindles.extend(current_frame_spindles)

    else:
        # compute the cost matrix as the Euclidean distance between centroids in the last frame and the current frame
        last_frame_spindles = [spindle for spindle in tracked_spindles if spindle['frame_number'] == frame_number - 1]
        last_frame_centroids = [spindle['centroid'] for spindle in last_frame_spindles]
        current_frame_centroids = [spindle['centroid'] for spindle in current_frame_spindles]
        cost_matrix = cdist(last_frame_centroids, current_frame_centroids)

        # use the Hungarian Algorithm to find the optimal assignment of spindles between frames
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # count the number of assignments to each spindle in the current frame
        assignment_counts = Counter(col_ind)

        # assign the spindles in the current frame to the spindles in the last frame
        for last_frame_index, current_frame_index in zip(row_ind, col_ind):
            # Get the spindles
            last_frame_spindle = last_frame_spindles[last_frame_index]
            current_frame_spindle = current_frame_spindles[current_frame_index]

            # check if the spindle has split, has disappeared, or is touching the image boundary
            if current_frame_spindle['area'] == 0 or \
               current_frame_spindle['bounding_box'][0] <= 0 or \
               current_frame_spindle['bounding_box'][1] <= 0 or \
               current_frame_spindle['bounding_box'][2] >= img_spindle_norm.shape[0] or \
               current_frame_spindle['bounding_box'][3] >= img_spindle_norm.shape[1] or \
               assignment_counts[current_frame_index] > 1:
                # if any of these conditions are true, don't assign it a tracked_spindle_number
                continue

            # if none of these conditions are true, assign it the same tracked_spindle_number as the last frame
            current_frame_spindle['tracked_spindle_number'] = last_frame_spindle['tracked_spindle_number']

        # add the spindles in the current frame to the list of all tracked spindles
        tracked_spindles.extend(current_frame_spindles)
        

    
# TODO: rename the output images (a combination of experiment name, time-stamp name etc ...)

# TODO: add overlay images as one of the output (spindle and cell cortex)

# TODO: add a csv file showing spindles and cells as one of the output

# TODO: use the brightfield channel to measure ellipticity change

# debug print    
time_elapsed = time.time() - since
print("Testing complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)) 

