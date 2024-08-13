# Multiple Spindles Tracking (Generalised from SpinX)
This repository is designed for hosting scripts for multi-instance tracking to generalise the current SpinX modules. <br/>

The scripts can be run in the command line.

For multiple spindle tracking
```python3 multispindle_tracker.py```
Running options:
--input_img: the input source image for nucleus counting (multi-stack tiff)
--time_stamp: define the start frame to track spindles, frame ID starting from 0, default set to 0
--nr_frames: define how many frames to track the movie
--spindle_channel: the spindle channel ID, starting from 0
--cell_channel: the cell (or brightfield) channel ID, starting from 0
--padding: define how many pixels to extend for each side of the bounding boxes to make them larger, default value set to 0
--output: define the output folder path
--auto_adjust: define whether to apply the auto-adjust function for low-intensity spindles, 'y' for apply. Be careful! When set this to 'y', other low-intensity non-spindle objects might also be detected.
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

For kinetochore tracking
```python3 kinetochore_tracker.py```
