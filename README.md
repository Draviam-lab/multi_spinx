# Multiple Spindles Tracking (Generalised from SpinX)
This repository is designed for hosting scripts for multi-instance tracking to generalise the current SpinX modules. The scripts can be run in the command line.

## Multiple spindle tracking
run```python3 multispindle_tracker.py``` <br/>

Running options: <br/>
--input_img: the input source image for nucleus counting (multi-stack tiff) <br/>
--time_stamp: define the start frame to track spindles, frame ID starting from 0, default set to 0 <br/>
--nr_frames: define how many frames to track the movie <br/>
--spindle_channel: the spindle channel ID, starting from 0 <br/>
--cell_channel: the cell (or brightfield) channel ID, starting from 0 <br/>
--padding: define how many pixels to extend for each side of the bounding boxes to make them larger, default value set to 0 <br/>
--output: define the output folder path <br/>
--auto_adjust: define whether to apply the auto-adjust function for low-intensity spindles, 'y' for apply. Be careful! When set this to 'y', other low-intensity non-spindle objects might also be detected. <br/>
--lower_marker: the lower marker for watershed segmentation, ranges from 0 to 1. <br/>
--higher_marker: The higher marker for watershed segmentation, ranges from 0 to 1. <br/>
--cropped: Whether export the cropped tracked-spindle images, 'y' for 'yes' all others for 'no'. <br/>

For kinetochore tracking
```python3 kinetochore_tracker.py``` <br/>

Running options: <br/>
--input_img: the input source image for nucleus counting (multi-stack tiff) <br/>
--time_stamp: define the start frame to track spindles, frame ID starting from 0, default set to 0 <br/>
--nr_frames: define how many frames to track the movie <br/>
--spindle_channel: the spindle channel ID, starting from 0 <br/>
--cell_channel: the GFP channel ID, starting from 0 <br/>
--padding: define how many pixels to extend for each side of the bounding boxes to make them larger, default value set to 0 <br/>
--output: define the output folder path <br/>
--auto_adjust: define whether to apply the auto-adjust function for low-intensity spindles, 'y' for apply. Be careful! When set this to 'y', other low-intensity non-spindle objects might also be detected. <br/>
--lower_marker: The lower marker for watershed segmentation, ranges from 0 to 1. <br/>
--higher_marker: The higher marker for watershed segmentation, ranges from 0 to 1. <br/>
--lower_marker_GFP: The lower marker for watershed segmentation for the GFP signals, ranges from 0 to 1. <br/>
--higher_marker_GFP: The higher marker for watershed segmentation for the GFP signals, ranges from 0 to 1. <br/>
--GFP_min_area: The min area of GFP signals in pixel, suggest to put 20. <br/>
--GFP_max_area: The max area of GFP signals in pixel, suggest to put a value less than 400. <br/>
--cropped: Whether export the cropped tracked-spindle images, 'y' for 'yes' all others for 'no' <br/>
