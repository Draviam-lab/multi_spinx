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

Parameters: 
    input_img: the input source image for nucleus counting (multi-stack tiff)
    
    time_stamp: define the start frame to track spindles, frame ID starting 
    from 1, default set to 1
    
    z_slice: the selected z-slice reference, starting from 1
    
    spindle_channel: the spindle channel ID, starting from 0
    
    cell_channel: the cell (or brightfield) channel ID, starting from 0
    
    padding: define how many pixels to extend for each side of the bounding boxes 
    to make them larger, default value set to 0
    
    output: define the output folder path
    
    nr_frames: define how many frames to track the movie
    
Returns: 
# TODO: TBC

"""

# package import
import time
import argparse
import warnings
import os
warnings.filterwarnings("ignore") # ignore warnings
since = time.time()

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import Counter

from skimage import segmentation, morphology
from skimage import io
from skimage.measure import regionprops
from skimage import transform

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

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
        # the time-stamp starts from 1, 
        # so when coding, need to use (opt.time_stamp-1) as the time-stamp reference
        "--time_stamp",
        type = int, 
        default = 1, 
        help = "define the start frame to track spindles, frame ID starting from 1, default set to 1" 
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


# TODO: these to be put within the tracking code
# source image read
# the sample image (stacked-tiff) is at size {TT, ZZ, XX, YY, CC}
# TT: time-stamp, ZZ: z-slice, XX and YY: size, CC: channels
img = io.imread(f"{opt.input_img}")
img_spindle = img[opt.time_stamp - 1, opt.z_slice - 1, :, :, opt.spindle_channel]
img_cell= img[opt.time_stamp - 1, opt.z_slice - 1, :, :, opt.cell_channel]
# normalisation to the [0, 1] scale for img_spindle and img_cell
img_spindle_norm = (img_spindle - img_spindle.min())/(img_spindle.max() - img_spindle.min())
img_cell_norm = (img_cell - img_cell.min())/(img_cell.max() - img_cell.min())



def spindle_segmentation(img):
    """
    This function segments the spindles using watershed method. The input of this
    function is a still image (in array of float64), and the outputs are the 
    segmented spindles (a bool/binary mask), the bounding box of each spindle
    (detected objects), the centroid coordinators and local centroid (relating 
    to the bounding box) of each spindle.
    
    Parameters: 
        Parameter 1: the normalised (at 0-1 scale) spindle image
        
    Returns: 
        Return 1: the segmented spindles (array of bool)
        Return 2: the list of bounding boxes for each spindle
        Return 3: the list of centroid for each spindle
        Return 4: the list of local centroid (relating to bounding box) for each spindle
    """
    # segmentation of the spindle(s) using the traditional watershed method
    # find the watershed markers of the background and the nuclei
    markers = np.zeros_like(img)
    markers[img < 0.3] = 1
    markers[img > 0.4] = 2
    # watershed segmentation of the spindles
    seg_spindle = segmentation.watershed(img, markers)
    seg_spindle = ndimage.binary_fill_holes(seg_spindle- 1)
    # remove small objects with boolean input "seg"
    seg_spindle = morphology.remove_small_objects(seg_spindle, 900)
        
    # generate spindle instance map based on the conventional watershed segmentation
    spindle_instance, nr_spindle = ndimage.label(seg_spindle)
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
        maxr, maxc = min(img_spindle.shape[0], maxr), min(img_spindle.shape[1], maxc)  
        
        # append the new bounding box to the list, 
        # only append the new bounding box if it does not touch the image boundary
        if minr > 0 and minc > 0 and maxr < img_spindle.shape[0] and maxc < img_spindle.shape[1]:
            bbox_list.append((minr, minc, maxr, maxc))
        
            # centroidarray coordinate tuple (row, col)
            centroid_list.append(spindle_regions[i].centroid)
            # centroid_local shows the centroid coordinate tuple (row, col), 
            # which is relative to region bounding box
            centroid_local_list.append(spindle_regions[i].centroid_local)
    
    # define the function returns
    return seg_spindle, bbox_list, centroid_list, centroid_local_list
    


    
seg_spindle, bbox_list, centroid_list, centroid_local_list = spindle_segmentation(img_spindle_norm)
  


# draw bounding box on top of the original images for illustration purpose
def bounding_box_plot(img, bbox_list):
    """
    This function plots the bounding box for illustration purpose
    Inputs: the source spindle image (single channel), bounding box list
    Output: the overlay image of bounding boxes on the original spindle image
    
    """
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
    
bounding_box_plot(img_spindle_norm, bbox_list)
bounding_box_plot(img_cell_norm, bbox_list)




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
    





# List to store the tracked spindles across all frames
tracked_spindles = []

# Process each frame
for frame_number in range(opt.time_stamp - 1, opt.time_stamp + opt.nr_frames - 1):
    # (Insert code here to extract the current frame and perform segmentation and bounding box generation)
    # TODO: need to do some encapsulation for the segmentation and bounding box generation to fit here
    
    # List to store the spindles in the current frame
    current_frame_spindles = []

    # Traverse the properties of each spindle
    for i in range(len(spindle_regions)):
        # (Insert code here to compute the bounding box and centroid of the spindle)

        # Compute the area of the bounding box
        area = (maxr - minr) * (maxc - minc)

        # Store the spindle in the current frame list
        current_frame_spindles.append({
            'frame_number': frame_number,
            'spindle_number': i,
            'bounding_box': (minr, minc, maxr, maxc),
            'centroid': (centroid_row, centroid_col),
            'area': area,
        })

    # If this is the first frame, just store the spindles without tracking
    if frame_number == 0:
        tracked_spindles.extend(current_frame_spindles)
    else:
        # Compute the cost matrix as the Euclidean distance between centroids in the last frame and the current frame
        last_frame_spindles = [spindle for spindle in tracked_spindles if spindle['frame_number'] == frame_number - 1]
        last_frame_centroids = [spindle['centroid'] for spindle in last_frame_spindles]
        current_frame_centroids = [spindle['centroid'] for spindle in current_frame_spindles]
        cost_matrix = cdist(last_frame_centroids, current_frame_centroids)

        # Use the Hungarian Algorithm to find the optimal assignment of spindles between frames
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Count the number of assignments to each spindle in the current frame
        assignment_counts = Counter(col_ind)

        # Assign the spindles in the current frame to the spindles in the last frame
        for last_frame_index, current_frame_index in zip(row_ind, col_ind):
            # Get the spindles
            last_frame_spindle = last_frame_spindles[last_frame_index]
            current_frame_spindle = current_frame_spindles[current_frame_index]

            # Check if the spindle has split, has disappeared, or is touching the image boundary
            if current_frame_spindle['area'] == 0 or \
               current_frame_spindle['bounding_box'][0] <= 0 or \
               current_frame_spindle['bounding_box'][1] <= 0 or \
               current_frame_spindle['bounding_box'][2] >= img_spindle.shape[0] or \
               current_frame_spindle['bounding_box'][3] >= img_spindle.shape[1] or \
               assignment_counts[current_frame_index] > 1:
                # If any of these conditions are true, don't assign it a tracked_spindle_number
                continue

            # If none of these conditions are true, assign it the same tracked_spindle_number as the last frame
            current_frame_spindle['tracked_spindle_number'] = last_frame_spindle['tracked_spindle_number']

        # Add the spindles in the current frame to the list of all tracked spindles
        tracked_spindles.extend(current_frame_spindles)

# Now the tracked_spindles list contains all spindles across all frames, with an additional 'tracked_spindle_number' field indicating the identity of the spindle across frames






    
# TODO: rename the output images (a combination of experiment name, time-stamp name etc ...)

# TODO: add overlay images as one of the output (spindle and cell cortex)

# TODO: add a csv file showing spindles and cells as one of the output

# TODO: can be optimised in the future by cropping the image with selected channel(s)
# with continuous time stamps. For example, if use the opt.time_stamp as the
# starting time stamp, and add a parameter opt.time_stamp_count standing for
# how many continuous time stamps should be taken into account. Might need to 
# use other channel's information to define when to start and when to stop the
# SpinX tracking.

# TODO: use the brightfield channel to measure ellipticity change





