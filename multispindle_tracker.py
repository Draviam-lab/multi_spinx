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
    Define the start frame to track spindles, frame ID starting from 0, default 
    set to 0.
    
nr_frames: int
    Define how many frames to track the movie.

spindle_channel: int
    The spindle channel ID, starting from 0.

cell_channel: int
    The cell (or brightfield) channel ID, starting from 0.

padding: int
    Define how many pixels to extend for each side of the bounding boxes to make 
    them larger, default value set to 0.

output: str
    Define the output folder path.

auto_adjust: str 
    Define whether to apply the auto-adjust function for low-intensity spindles 
    'y' for apply. Be careful! When set this to 'y', other low-intensity
    non-spindle objects/structures might also be detected.
    
lower_marker: float
    The lower marker for watershed segmentation, ranges from 0 to 1.
    
higher_marker: float
    The higher marker for watershed segmentation, ranges from 0 to 1.
    
cropped: str
    Define whether export the cropped tracked-spindle images, 'y' for 'yes' and 
    all others for 'no'.
    
Returns
-------
cropped_images (folder): 
    contains two subfolders, storing the cropped spindles and cells respectively.
    
cropped_images_rescaled_to_450_450 (folder): 
    contains two subfolders, storing the cropped spindle and cells respectively 
    with them rescale to 450*450.
    
tracked_spindles_summary_frame_X_to_Y.csv:  
    this spreadsheet shows the summary for all tracked spindles from the tracked 
    frame X to frame Y.
    
tracked_spindles_summary_frame_X_to_Y.tif: 
    this is a multi-stacked tiff overlay result showing spindle tracking from 
    frame X to frame Y, spindles are highlighted with bounding boxes and are 
    labelled with a unique number as the identifier. This file can be opened 
    with ImageJ/Fiji.

"""

# package import
import time
import argparse
import warnings
import os
import re

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
# from collections import Counter

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
        default = "F:/Dropbox/Postdoc_QMUL/workspace/multispindle/data/exp2022_H1299_pi-EB1-GFP_EB3-mKate2_SiR-DNA_set21_DMSO-1-5_CilioDi-5uM-6-10_1_10_R3D.tif", 
        help = "the input source image for nucleus counting (multi-stack tiff)" 
        )
    parser.add_argument(
        # the time-stamp starts from 0,
        # so if start from time frame t in the movie, then here should be (t - 1)
        "--time_stamp",
        type = int, 
        default = 0, 
        help = "define the start frame to track spindles, frame ID starting from 0, default set to 0" 
        )
    parser.add_argument(
        "--nr_frames",
        type = int, 
        default = 49, 
        help = "define how many frames to track the movie" 
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
        default = "F:/Dropbox/Postdoc_QMUL/workspace/multispindle/output/results", 
        help = "define the output folder path" 
        )
    parser.add_argument(
        "--auto_adjust",
        type = str, 
        default = "n", 
        help = "Whether to apply the auto-adjust function for low-intensity spindles \
            'y' for apply. Be careful! When set this to 'y', other low-intensity \
            non-spindle objects might also be detected." 
        )
    parser.add_argument(
        "--lower_marker",
        type = float, 
        default = 0.30, 
        help = "The lower marker for watershed segmentation, ranges from 0 to 1." 
        )
    parser.add_argument(
        "--higher_marker",
        type = float, 
        default = 0.40, 
        help = "The higher marker for watershed segmentation, ranges from 0 to 1." 
        )
    parser.add_argument(
        "--cropped",
        type = str, 
        default = "y", 
        help = "Whether export the cropped tracked-spindle images, 'y' for 'yes' all others for 'no'" 
        )

    opt = parser.parse_args()
    print(opt)

# check whether output folder exist
os.makedirs(f"{opt.output}", exist_ok = True)
# extract the basename (i.e., filename with extension)
filename_with_extension = os.path.basename(opt.input_img)
# use regular expression to extract the desired part
filename = re.match(r"^(.*?)\.[^.]*$", filename_with_extension).group(1) 
# check subfolder for output
os.makedirs(f"{opt.output}/{filename}", exist_ok = True)   

def img_read(img_path, time_stamp, spindle_channel, cell_channel):
    """
    This function operates on the multi-stacked tiff image (movie) at 
    {TT, ZZ, XX, YY, CC} structure, where TT stands for time-stamp, ZZ stands 
    for z-slice, XX and YY stand for the size at each frame, and CC stands for
    channels. This function reads the specific image channels for a specific 
    time frame and applies a maximisation projection across all z-slices. 
    
    Parameters
    ----------
    img_path: str
        The input source image for nucleus counting (multi-stack tiff).
        
    time_stamp: int
        Define the start frame to track spindles, frame ID starting from 1, 
        default set to 1.
        
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

    # the sample image (stacked-tiff) is at {TT, ZZ, XX, YY, CC} structure
    img = imread(img_path) # source image read

    # selecting specific time-stamp and channel, 
    # and then applying maximisation projection over all the z-slices
    img_spindle = np.max(img[time_stamp, :, :, :, spindle_channel], axis = 0)
    img_cell = np.max(img[time_stamp, :, :, :, cell_channel], axis = 0)
    
    # normalisation to the [0, 1] scale for img_spindle and img_cell
    img_spindle_norm = (img_spindle - img_spindle.min())/(img_spindle.max() - img_spindle.min())
    img_cell_norm = (img_cell - img_cell.min())/(img_cell.max() - img_cell.min())
    
    return img_spindle_norm.astype(np.float16), img_cell_norm.astype(np.float16)

def auto_adjust(img_norm):
    """
    This function is a Python-rewriting of ImageJ's auto-threshold option
    (Image > Adjust > Brightness/Contrast > 'Auto' button)
    Based on https://github.com/imagej/ImageJ/blob/706f894269622a4be04053d1f7e1424094ecc735/ij/plugin/frame/ContrastAdjuster.java#L780
    
    The algorithm (function autoAdjust) is basically a contrast setting the max 
    white value to the max of the image (and same for black for min), with some 
    saturation : i.e., it's not the max(min) of the image which is actually used 
    but a lower(higher) value to eliminate the thin "tails" of the histogram 
    and get an output dynamic range which allows for good visualisation of most 
    of the image's pixels (at the expense of a few saturated pixels). While some 
    (or most) algorithms parametrize this saturation to eliminate a set percentage 
    of pixels, ImageJ's algorithm selects the closest values to the max(min) 
    values whose count are over a certain proportion of the total amount of pixels.
    
    Parameters
    ----------
    img_norm: ndarray (2D)
        Data array stands for the the normalised (0-1 scale) spindle image.
        
    Returns
    -------
    imr: ndarray of bool (2D)
        Data array stands for the the normalised (0-1 scale) spindle image after
        the auto-adjust processing.
    """
    
    im_min = np.min(img_norm)
    im_max = np.max(img_norm)
    
    # histogram computation 
    hist_min = im_min
    hist_max = im_max
    histogram = np.histogram(img_norm, bins = 256, range = (hist_min, hist_max))[0]
    bin_size = (hist_max - hist_min) / 256

    # compute output min and max bins 
    h, w = img_norm.shape
    pixel_count = h * w
    # the following values are taken directly from the ImageJ file.
    limit = pixel_count/10
    const_auto_threshold = 5000
    auto_threshold = 0

    auto_threshold = const_auto_threshold if auto_threshold <= 10 else auto_threshold/2
    threshold = int(pixel_count/auto_threshold)

    # setting the output min bin
    i = -1
    found = False
    # going through all bins of the histogram in increasing order until you reach one where the count if more than
    # pixel_count/auto_threshold
    while not found and i <= 255:
        i += 1
        count = histogram[i]
        if count > limit:
            count = 0
        found = count > threshold
    hmin = i
    found = False

    # setting the output max bin : same thing but starting from the highest bin.
    i = 256
    while not found and i > 0:
        i -= 1
        count = histogram[i]
        if count > limit:
            count = 0
        found = count > threshold
    hmax = i

    # compute output min and max pixel values from output min and max bins 
    if hmax >= hmin:
        min_ = hist_min + hmin * bin_size
        max_ = hist_min + hmax * bin_size
        # bad case number one, just return the min and max of the histogram
        if min_ == max_:
            min_ = hist_min
            max_ = hist_max
    # bad case number two, same
    else:
        min_ = hist_min
        max_ = hist_max

    # apply the contrast 
    imr = (img_norm-min_) / (max_-min_)
    
    return imr

def spindle_segmentation(img, lower_marker, higher_marker):
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
    markers[img < lower_marker] = 1
    markers[img > higher_marker] = 2
    # watershed segmentation of the spindles
    seg_spindle = watershed(img, markers)
    seg_spindle = binary_fill_holes(seg_spindle - 1)
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

def bounding_box_plot(img, bbox_list):
    """
    This function plots the bounding box for single frame on selected channel
    for illustration purpose.

    Parameters
    ----------
    img: ndarray (2D)
        Data array stands for the the normalised (0-1 scale) spindle image.
        
    bbox_list: list
        The list of bounding boxes (min_row, min_col, max_row, max_col) for 
        each detected spindle.

    Returns
    -------
    Currently none, the function only makes the plot for single frame on 
    selected channel.
    """
    
    import matplotlib.patches as mpatches
    
    # define the figure and plot the original image
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.imshow(img)
    
    # draw bounding boxes accordingly on the original image
    for bboxes in bbox_list:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = bboxes
        rect = mpatches.Rectangle(
            (minc, minr), maxc - minc, maxr - minr, 
            fill = False, edgecolor = 'red', linewidth = 2)
        ax.add_patch(rect)
    
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()  

def bounding_box_plot_5d(img_path, output_path, nr_frame, bbox_list_per_time, channel, start_frame):
    """
    This function plots the bounding boxes on the maximisation projection 
    across all the z-slices for each time point. The overlay images will then 
    stacked as a multi-stacked tiff file as the output.

    Parameters
    ----------
    img_5d: ndarray
        The 5D multi-stacked TIFF image with dimensions {TT, ZZ, XX, YY, CC}.
        
    bbox_list_per_time: list of list
        A list containing bounding box lists for each time point.
        
    channel: int
        The channel ID to visualise (i.e., brightfield or spindle channel).
        
    start_frame: int
        The starting frame for tracking.
    """
    
    import matplotlib.patches as mpatches
    from skimage.io import imread, imsave

    # the sample image (stacked-tiff) is at {TT, ZZ, XX, YY, CC} structure
    img_5d = imread(img_path) # source image read
    
    # define number of frames to track 
    num_time_points = nr_frame # nr_frame should be opt.nr_frames or img_5d.shape[0]
    
    output_images = [] # create an empty list output_images before the loop
    
    for t in range(num_time_points):
        # maximisation projection across z-slices for the current time point 
        # on specified channel
        # (t + start_frame) stand for the relative frame ID if not start from frame 0
        max_projected_img = np.max(img_5d[t + start_frame, :, :, :, channel], axis = 0)
        
        # define new figure size
        # desired figure size in pixels
        width_px, height_px = np.shape(max_projected_img)
        dpi = 100  # set DPI
        width_in = width_px / dpi
        height_in = height_px / dpi
        
        # define the figure and plot the original image
        fig, ax = plt.subplots(figsize = (width_in, height_in), dpi = dpi)
        ax.imshow(max_projected_img, cmap = 'gray')
        # ax.imshow(max_projected_img)
        
        # (t + start_frame) stand for the relative frame ID if not start from frame 0
        tracked_spindles_at_frame = [spindle for spindle in tracked_spindles if spindle.get('frame_number') == (t + start_frame)]
        
        # plotting the bounding boxes for the current time point
        for i in range(len(tracked_spindles_at_frame)):
            # draw bounding box
            minr, minc, maxr, maxc = tracked_spindles_at_frame[i]['bounding_box']
            rect = mpatches.Rectangle(
                (minc, minr), maxc - minc, maxr - minr, 
                fill = False, edgecolor = 'red', linewidth = 2)
            ax.add_patch(rect)
            # draw text along with the bounding box to identify spindle_id
            centroid_y, centroid_x = tracked_spindles_at_frame[i]['centroid']
            spindle_id = tracked_spindles_at_frame[i]['tracked_spindle_number']
            if spindle_id != None:
                ax.text(
                    minc + 5, minr + 25,
                    # centroid_x, centroid_y, 
                    str(spindle_id), 
                    color = 'red', fontsize = 18
                    )
            elif spindle_id == None:
                ax.text(
                    minc + 5, minr + 25,
                    # centroid_x, centroid_y, 
                    "new",
                    color = 'red', fontsize = 18
                    )
            
        # capture the figure's image data without displaying it
        ax.set_axis_off()
        plt.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, wspace = 0, hspace = 0)
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())
        output_images.append(data)
        
        plt.close(fig)
        # plt.show()
    
    # save the images as a multi-stacked TIFF file
    imsave(output_path, np.array(output_images))

def spindles_to_csv(output_path, tracked_spindles):
    """
    This function converts the tracked_spindles to a csv file. The tracked_spindles 
    should be a list of dictionary, and in the dictionary there are fields of
    area (float), bounding_box (tuple), centroid (tuple), frame_number (int), 
    spindle_number (int) and tracked_spindle_number (int).

    Parameters
    ----------
    tracked_spindles : list (list of dictionary)
        A list to store the tracked spindles across all frames, with an 
        additional tracked_spindle_number field indicating the identity of 
        the spindle across frames.
        
    output_path: str
        The output path of the output csv file.

    Returns
    -------
    This function will return and save a csv file containing tracked spindles 
    information on selected path.
    
    """
    
    df = pd.DataFrame(tracked_spindles)

    # sort the dataframe respectively for the tracked_spindle_number is or is not None
    df_with_number = df[df['tracked_spindle_number'].notna()]
    df_without_number = df[df['tracked_spindle_number'].isna()]

    df_with_number = df_with_number.sort_values(by=['tracked_spindle_number', 'frame_number'])
    df_without_number = df_without_number.sort_values(by=['frame_number', 'spindle_number'])

    sorted_df = pd.concat([df_with_number, df_without_number])

    # extract min_row, min_col, max_row, max_col from bounding_box
    # extract centroid_row and centroid_col from centroid
    sorted_df[['min_row', 'min_col', 'max_row', 'max_col']] = pd.DataFrame(
        sorted_df['bounding_box'].tolist(), 
        index = sorted_df.index
        )
    sorted_df[['centroid_row', 'centroid_col']] = pd.DataFrame(
        sorted_df['centroid'].tolist(), 
        index = sorted_df.index
        )
    # drop the bounding_box and centroid columns
    sorted_df = sorted_df.drop(columns = ['bounding_box', 'centroid'])
    
    # fix the starting number of frame (fix from starting from 0 to starting from 1)
    sorted_df['frame_number'] = sorted_df['frame_number'] + 1

    # write to csv
    sorted_df.to_csv(
        output_path, 
        columns = ['tracked_spindle_number', 'frame_number', 'min_row', 'min_col', 'max_row', 'max_col', 'centroid_row', 'centroid_col'], 
        index = False
        )

########## below code are the main flow for multi-spindle tracking ##########
    
# list to store the tracked spindles across all frames,
# with an additional tracked_spindle_number field indicating the identity of 
# the spindle across frames.
tracked_spindles = []
# create another list of list stands for the list of the bounding boxes list 
# across time frame
bbox_list_per_time = []
# define the spindle ID
next_spindle_id = 1 
# process each frame
# frame_number here is not the absolute frame_number of the multi-stacked tiff
# but the relative frame_number in the [start_time_stamp - 1, end_time_stamp) range.
for frame_number in range(opt.time_stamp, opt.time_stamp + opt.nr_frames):    
    # image read for the current frame,
    # the spindle and cell cortex channels are both normalised
    img_spindle_norm, img_cell_norm = img_read(
        f"{opt.input_img}", 
        frame_number, 
        opt.spindle_channel, 
        opt.cell_channel)
    
    if opt.auto_adjust == "y":
        img_spindle_norm = auto_adjust(img_spindle_norm)
    
    # perform spindle segmentation and bounding box generation for the current frame
    seg_spindle, bbox_list, centroid_list, _ = spindle_segmentation(
        img_spindle_norm, opt.lower_marker, opt.higher_marker
        )
    
    bbox_list_per_time.append(bbox_list)
    
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
    if frame_number == opt.time_stamp:
        for i, spindle in enumerate(current_frame_spindles):
            spindle['tracked_spindle_number'] = next_spindle_id
            next_spindle_id = next_spindle_id + 1
        tracked_spindles.extend(current_frame_spindles)

    else:
        # extract the tracked spindles information at their latest appearance 
        # for all the tracked spindles 
        latest_spindle_summary = {}
        for spindle in tracked_spindles:
            tracked_id = spindle['tracked_spindle_number']
            frame_id = spindle['frame_number']
            if tracked_id == None:
                continue
            elif tracked_id not in latest_spindle_summary or frame_id > latest_spindle_summary[tracked_id]['frame_number']:
                latest_spindle_summary[tracked_id] = spindle
        all_tracked_spindles_summary = [value for key, value in latest_spindle_summary.items()]
        all_tracked_spindles_centroids = [spindle['centroid'] for spindle in all_tracked_spindles_summary]
        
        # define the spindles in the last frame
        last_frame_spindles = [spindle for spindle in tracked_spindles if spindle['frame_number'] == frame_number - 1]
        last_frame_centroids = [spindle['centroid'] for spindle in last_frame_spindles]
        
        # define the centroids of spindles for the current frams
        current_frame_centroids = [spindle['centroid'] for spindle in current_frame_spindles]
        
        try:
            ########## Strategy 1 ##########
            # when use this strategy, the other strategy should be commented out
            # A computation between the all tracked spindles (summary) and the current
            # frame should be calculated
            cost_matrix = cdist(all_tracked_spindles_centroids, current_frame_centroids)
    
            # use the Hungarian Algorithm to find the optimal assignment of spindles between frames
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Assuming the following structures:
            # all_tracked_spindles_centroids: list of centroids (tuples) for all previously tracked spindles
            # current_frame_centroids: list of centroids (tuples) for spindles in the current frame
            # next_spindle_id: an integer tracking the next available spindle ID
            
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < 80: # distance_threshold = 80
                    # Spindle is considered the same; assign existing tracked_spindle_number
                    current_frame_spindles[j]['tracked_spindle_number'] = all_tracked_spindles_summary[i]['tracked_spindle_number']
                else:
                    # Spindle is new; assign a new tracked_spindle_number
                    current_frame_spindles[j]['tracked_spindle_number'] = next_spindle_id
                    next_spindle_id += 1
        
            # Handle any completely new spindles not matched in the cost matrix
            for spindle in current_frame_spindles:
                if 'tracked_spindle_number' not in spindle or spindle['tracked_spindle_number'] is None:
                    spindle['tracked_spindle_number'] = next_spindle_id
                    next_spindle_id += 1
                    
            # add the spindles in the current frame to the list of all tracked spindles
            tracked_spindles.extend(current_frame_spindles)
            
            ########## Strategy 2 ##########
            # # when use this strategy, the other strategy should be commented out
            # # Computed the cost matrix as the Euclidean distance between centroids 
            # # in the last frame and the current frame 
            # cost_matrix = cdist(last_frame_centroids, current_frame_centroids)
            
            # # use the Hungarian Algorithm to find the optimal assignment of spindles between frames
            # row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # # count the number of assignments to each spindle in the current frame
            # assignment_counts = Counter(col_ind)
    
            # # assign the spindles in the current frame to the spindles in the last frame
            # for last_frame_index, current_frame_index in zip(row_ind, col_ind):
            #     # Get the spindles
            #     last_frame_spindle = last_frame_spindles[last_frame_index]
            #     current_frame_spindle = current_frame_spindles[current_frame_index]
    
            #     # check if the spindle has split, has disappeared, or is touching the image boundary
            #     if current_frame_spindle['area'] == 0 or \
            #         current_frame_spindle['bounding_box'][0] <= 0 or \
            #         current_frame_spindle['bounding_box'][1] <= 0 or \
            #         current_frame_spindle['bounding_box'][2] >= img_spindle_norm.shape[0] or \
            #         current_frame_spindle['bounding_box'][3] >= img_spindle_norm.shape[1] or \
            #         assignment_counts[current_frame_index] > 1:
            #         # if any of these conditions are true, don't assign it a tracked_spindle_number
            #         continue
    
            #     # if none of these conditions are true, assign it the same tracked_spindle_number as the last frame
            #     if last_frame_spindle['tracked_spindle_number'] != None:
            #         current_frame_spindle['tracked_spindle_number'] = last_frame_spindle['tracked_spindle_number']
            #     else:
            #         current_frame_spindle['tracked_spindle_number'] = next_spindle_id
            #         next_spindle_id = next_spindle_id + 1
              
            # # add the spindles in the current frame to the list of all tracked spindles
            # tracked_spindles.extend(current_frame_spindles)
    
        except:
            pass
        
    # debug print
    print(f"frame {frame_number + 1} complete")

# output the tracked_spindles in a csv file
spindles_to_csv(
    f"{opt.output}/{filename}/tracked_spindles_summary_frame_{frame_number + 1 - opt.nr_frames + 1}_to_{frame_number + 1}.csv", 
    tracked_spindles
    )
# output the overlay multi-stacked tiff file
bounding_box_plot_5d(
    f"{opt.input_img}", 
    f"{opt.output}/{filename}/tracked_spindles_summary_frame_{frame_number + 1 - opt.nr_frames + 1}_to_{frame_number + 1}.tif", 
    opt.nr_frames, 
    bbox_list_per_time, 
    opt.spindle_channel,
    opt.time_stamp
    )

if opt.cropped == "y":
    # output two folders containing the cropped spindles
    # one folder contains the original images, and the other folder contains the rescaled images
    # each of the folders will have two subfolders, containing the crops of spindles and cells
    # check if four sub-folders exist
    os.makedirs(f"{opt.output}/{filename}/cropped_images/spindle", exist_ok = True)
    os.makedirs(f"{opt.output}/{filename}/cropped_images/cell", exist_ok = True)
    os.makedirs(f"{opt.output}/{filename}/cropped_images_rescaled_to_450_450/spindle", exist_ok = True)
    os.makedirs(f"{opt.output}/{filename}/cropped_images_rescaled_to_450_450/cell", exist_ok = True)
    
    for spindle in tracked_spindles:
        # extract tracked spindle information
        minr, minc, maxr, maxc = spindle["bounding_box"]
        # the starting value of frame_number in img_read() function is 1 
        # where in tracked_spindles is from 0, so the second peremeter should +1
        frame_id = spindle["frame_number"] + 1
        spindle_id = spindle["tracked_spindle_number"]
        
        # extract frame
        img_spindle_norm, img_cell_norm = img_read(
            f"{opt.input_img}", 
            spindle["frame_number"], 
            opt.spindle_channel, 
            opt.cell_channel
            )
        
        # image operation on the normalised channels
        cropped_spindle = img_spindle_norm[int(minr): int(maxr), int(minc): int(maxc)]
        cropped_cell = img_cell_norm[int(minr): int(maxr), int(minc): int(maxc)]
        
        # save the cropped images as single-channel .tiff files
        io.imsave(
            f"{opt.output}/{filename}/cropped_images/spindle/cropped_frame_{frame_id}_spindle_{spindle_id}.tif",
            cropped_spindle
            )
        io.imsave(
            f"{opt.output}/{filename}/cropped_images/cell/cropped_frame_{frame_id}_spindle_{spindle_id}.tif", 
            cropped_cell
            )
        
        # resize the cropped images to the desired output size
        output_size = (450, 450) # define the size of the output images
        resized_spindle = transform.resize(cropped_spindle, output_size)
        resized_cell = transform.resize(cropped_cell, output_size)
        
        # TODO: the cropped images should be at the same scale for the same spindle (across time)
        # even if the bounding boxes of the same spindle in different time frame is at
        # different size.. One way to solve this is to fix the bounding box size for each
        # of the spindle (eg. use the first bounding box at the first frame), and the 
        # move the bounding box according with the centroid.
    
        # save the cropped and rescaled images as single-channel .tiff files
        io.imsave(
            f"{opt.output}/{filename}/cropped_images_rescaled_to_450_450/spindle/cropped_rescaled_frame_{frame_id}_spindle_{spindle_id}.tif",
            resized_spindle
            )
        io.imsave(
            f"{opt.output}/{filename}/cropped_images_rescaled_to_450_450/cell/cropped_rescaled_frame_{frame_id}_spindle_{spindle_id}.tif", 
            resized_cell
            )

# debug print    
time_elapsed = time.time() - since
print("Testing complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)) 

