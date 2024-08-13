# Multiple Spindles Tracking (Generalised from SpinX)
This repository is designed for hosting scripts for multi-instance tracking to generalise the current SpinX modules.

The scripts can be run in the command line.

For multiple spindle tracking
```python3 multispindle_tracker.py```

Running options:

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

For kinetochore tracking
```python3 kinetochore_tracker.py```
