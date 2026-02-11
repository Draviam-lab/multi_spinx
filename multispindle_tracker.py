# -*- coding: utf-8 -*-
"""
CLI entrypoint for multi-spindle tracking on 5D microscopy TIFF movies.

This script keeps the original command-line behavior and tracking flow, while
delegating reusable image/segmentation/visualization/export logic to the
`tracking_core` package.

Author: Dr Binghao Chai
Institute: Queen Mary University of London

Main responsibilities:
1. Parse CLI arguments and create output folders.
2. Run frame-by-frame spindle detection and ID assignment.
3. Export CSV summaries, overlay TIFF stacks, and optional crop datasets.

CLI usage (argparse):
    python3 multispindle_tracker.py --help
    python3 multispindle_tracker.py --input_img <movie.tif> --output <output_dir> [options]

CLI arguments:
- --input_img: source multi-stack TIFF path.
- --time_stamp: zero-based start frame (default: 0).
- --nr_frames: number of frames to process (default: 49).
- --spindle_channel: spindle channel index (default: 0).
- --cell_channel: brightfield/cell channel index (default: 3).
- --padding: bbox padding in pixels (default: 40).
- --output: output directory path.
- --auto_adjust: apply ImageJ-like auto-contrast, use 'y' to enable (default: 'n').
- --lower_marker: watershed lower marker (default: 0.30).
- --higher_marker: watershed higher marker (default: 0.40).
- --cropped: export crops, use 'y' to enable (default: 'y').

Notes:
- `time_stamp` is zero-based.
- This script expects channel-last indexing for the source image:
  `{T, Z, X, Y, C}`.
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

from tracking_core.exporters import spindles_to_csv as core_spindles_to_csv
from tracking_core.image_ops import auto_adjust as core_auto_adjust
from tracking_core.image_ops import read_projected_channels
from tracking_core.segmentation import spindle_segmentation as core_spindle_segmentation
from tracking_core.visualization import bounding_box_plot as core_bounding_box_plot
from tracking_core.visualization import write_tracking_overlay_tiff

warnings.filterwarnings("ignore") # ignore warnings
since = time.time()

# arguments definition
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input_img",
        type = str, 
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
    Backward-compatible wrapper for shared 5D TIFF channel reading.

    Delegates to `tracking_core.image_ops.read_projected_channels` with
    channel-last indexing (`{T, Z, X, Y, C}`).
    """
    return read_projected_channels(
        img_path=img_path,
        time_stamp=time_stamp,
        spindle_channel=spindle_channel,
        cell_channel=cell_channel,
        channel_axis_last=True,
    )


def auto_adjust(img_norm):
    """Backward-compatible wrapper for ImageJ-style auto contrast."""
    return core_auto_adjust(img_norm)


def spindle_segmentation(img, lower_marker, higher_marker):
    """
    Backward-compatible wrapper for spindle segmentation.

    Keeps script-level `opt.padding` behavior unchanged.
    """
    return core_spindle_segmentation(
        img=img,
        lower_marker=lower_marker,
        higher_marker=higher_marker,
        padding=opt.padding,
    )


def bounding_box_plot(img, bbox_list):
    """Plot one frame with bounding boxes for interactive inspection."""
    return core_bounding_box_plot(img=img, bbox_list=bbox_list, linewidth=2)


def bounding_box_plot_5d(
    img_path, output_path, nr_frame, bbox_list_per_time, channel, start_frame
):
    """
    Write frame overlays as a multi-stacked TIFF for tracked spindles.

    The `bbox_list_per_time` argument is accepted for compatibility and is not
    used directly, matching prior behavior.
    """
    return write_tracking_overlay_tiff(
        img_path=img_path,
        output_path=output_path,
        nr_frame=nr_frame,
        channel=channel,
        start_frame=start_frame,
        tracked_items=tracked_spindles,
        tracked_id_key="tracked_spindle_number",
        channel_axis_last=True,
    )


def spindles_to_csv(output_path, tracked_spindles):
    """Export tracked spindle records to the legacy CSV schema."""
    return core_spindles_to_csv(output_path=output_path, tracked_spindles=tracked_spindles)

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

