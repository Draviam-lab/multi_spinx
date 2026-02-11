# Multi-SpinX

Multi-SpinX provides command-line scripts for automated tracking of mitotic spindles and kinetochores (GFP) in microscopy movies.

Author: Dr Binghao Chai  
Institute: Queen Mary University of London

## Related publication

Paper link: https://www.sciencedirect.com/science/article/pii/S0010482524017116

## Repository structure

Current top-level structure:

```text
multi_spinx/
├── README.md
├── LICENSE
├── .gitignore
├── install_conda_env.sh
├── multispindle_tracker.py
├── kinetochore_tracker.py
├── tracking_core/
│   ├── __init__.py
│   ├── image_ops.py
│   ├── segmentation.py
│   ├── visualization.py
│   └── exporters.py
├── analysis/
│   ├── bounding_box_change_area.py
│   └── bounding_box_change_percentage.py
└── assessment_form/
    ├── multispindle_tracker/
    └── kinetochore_tracker/
```

## Installation

This repository provides a single-file, one-click Conda installer:

- `install_conda_env.sh`

The script was prepared from import checks across all Python scripts in this
repository and installs the required third-party packages:

- `numpy`
- `pandas`
- `scipy`
- `scikit-image`
- `matplotlib`

Supported systems:

- macOS
- Linux

Steps:

1. Open a terminal in the repository root.
2. Run:

```bash
./install_conda_env.sh
```

3. Activate environment:

```bash
conda activate multi_spinx
```

Optional: custom environment name

```bash
./install_conda_env.sh my_env_name
conda activate my_env_name
```

Optional: custom Python version (default is `3.11`)

```bash
PYTHON_VERSION=3.10 ./install_conda_env.sh
```

## Script overview

### `multispindle_tracker.py`

Tracks multiple spindles across frames and exports:

- spindle tracking CSV summary
- spindle overlay TIFF stack
- optional cropped spindle/cell images (original and resized to 450x450)

Image layout expectation for this script:

- `{T, Z, X, Y, C}` (channel-last)

### `kinetochore_tracker.py`

Tracks both spindles and GFP (kinetochores) across frames and exports:

- spindle tracking CSV summary
- GFP tracking CSV summary
- spindle overlay TIFF stack (spindle channel)
- spindle overlay TIFF stack (GFP channel)
- GFP overlay TIFF stack
- GFP mask TIFF stack
- optional cropped spindle images (original and resized to 450x450)

Image layout expectation for this script:

- `{T, Z, C, X, Y}` (channel-first)

### `tracking_core/` (shared sub-package)

Shared logic extracted from tracker scripts:

- `image_ops.py`: 5D TIFF channel reading, max projection, normalization, ImageJ-style auto-adjust
- `segmentation.py`: spindle and GFP segmentation utilities
- `visualization.py`: single-frame bbox plotting and overlay TIFF export
- `exporters.py`: CSV export formatting for spindle/GFP tracking tables

### `analysis/` scripts

Post-hoc plotting helpers:

- `analysis/bounding_box_change_area.py`: plot absolute bounding-box area over time
- `analysis/bounding_box_change_percentage.py`: plot normalized area change over time

These scripts are intended for ad-hoc analysis and visualization, not as primary tracking entrypoints.

## CLI usage (argparse)

### Common argparse usage pattern

Show help:

```bash
python3 <script>.py --help
```

Run with required runtime inputs:

```bash
python3 <script>.py --input_img <movie.tif> --output <output_dir> [options]
```

Note: both scripts define many defaults, but in normal use you should provide `--input_img` and `--output`.

## `multispindle_tracker.py` arguments

```bash
python3 multispindle_tracker.py --help
```

- `--input_img` (str): input multi-stack TIFF path
- `--time_stamp` (int, default `0`): zero-based start frame
- `--nr_frames` (int, default `49`): number of frames to process
- `--spindle_channel` (int, default `0`): spindle channel index
- `--cell_channel` (int, default `3`): brightfield/cell channel index
- `--padding` (int, default `40`): pixel padding for square bounding boxes
- `--output` (str): output directory
- `--auto_adjust` (str, default `"n"`): set `"y"` to enable auto contrast
- `--lower_marker` (float, default `0.30`): watershed lower marker
- `--higher_marker` (float, default `0.40`): watershed higher marker
- `--cropped` (str, default `"y"`): set `"y"` to export crops

Example:

```bash
python3 multispindle_tracker.py \
  --input_img movie.tif \
  --output outputs \
  --time_stamp 0 \
  --nr_frames 49 \
  --spindle_channel 0 \
  --cell_channel 3
```

## `kinetochore_tracker.py` arguments

```bash
python3 kinetochore_tracker.py --help
```

- `--input_img` (str): input multi-stack TIFF path
- `--time_stamp` (int, default `2`): zero-based start frame
- `--nr_frames` (int, default `26`): number of frames to process
- `--spindle_channel` (int, default `1`): spindle channel index
- `--cell_channel` (int, default `0`): GFP channel index
- `--padding` (int, default `40`): pixel padding for square bounding boxes
- `--output` (str): output directory
- `--auto_adjust` (str, default `"n"`): set `"y"` to enable auto contrast
- `--lower_marker` (float, default `0.15`): spindle watershed lower marker
- `--higher_marker` (float, default `0.25`): spindle watershed higher marker
- `--lower_marker_GFP` (float, default `0.15`): GFP watershed lower marker
- `--higher_marker_GFP` (float, default `0.28`): GFP watershed higher marker
- `--GFP_min_area` (int, default `20`): GFP minimum area threshold
- `--GFP_max_area` (int, default `300`): GFP maximum area threshold
- `--cropped` (str, default `"y"`): set `"y"` to export crops

Example:

```bash
python3 kinetochore_tracker.py \
  --input_img movie.tif \
  --output outputs \
  --time_stamp 2 \
  --nr_frames 26 \
  --spindle_channel 1 \
  --cell_channel 0
```

## Output organization

Each run creates an output subfolder named after the input TIFF filename stem:

```text
<output>/<input_name_without_extension>/
```

Within that folder, file names include the processed frame range:

- `tracked_spindles_summary_frame_X_to_Y.csv`
- `tracked_spindles_frame_X_to_Y.tif`
- `tracked_spindles_GFP_channel_frame_X_to_Y.tif` (kinetochore script)
- `tracked_gfp_summary_frame_X_to_Y.csv` (kinetochore script)
- `tracked_GFPs_frame_X_to_Y.tif` (kinetochore script)
- `GFP_masks_frame_X_to_Y.tif` (kinetochore script)
- crop folders when `--cropped y`

## Citation

If you use this tool, please cite:

```bibtex
@article{chai2025multi,
  title={Multi-SpinX: An advanced framework for automated tracking of mitotic spindles and kinetochores in multicellular environments},
  author={Chai, Binghao and Efstathiou, Christoforos and Choudhury, Muntaqa S and Kuniyasu, Kinue and Jain, Saakshi Sanjay and Maharea, Alexia-Cristina and Tanaka, Kozo and Draviam, Viji M},
  journal={Computers in Biology and Medicine},
  volume={186},
  pages={109626},
  year={2025},
  publisher={Elsevier}
}
```

## License

This project is licensed under the Apache License 2.0.

See `LICENSE` for the full license text and terms.

## useful link

1. Draviam Lab: http://www.draviamlab.uk/
2. Dr Binghao Chai: https://bhchai.com/
