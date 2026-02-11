"""
Shared tracking helpers extracted from legacy CLI scripts.

The public API in this package is intentionally small and maps to the core
operations used by both tracker entrypoints:
- channel extraction and normalization
- segmentation
- visualization overlays
- CSV export
"""

from .exporters import gfps_to_csv, spindles_to_csv
from .image_ops import auto_adjust, read_projected_channels
from .segmentation import gfp_segmentation, spindle_segmentation
from .visualization import bounding_box_plot, write_tracking_overlay_tiff

__all__ = [
    "auto_adjust",
    "bounding_box_plot",
    "gfp_segmentation",
    "gfps_to_csv",
    "read_projected_channels",
    "spindle_segmentation",
    "spindles_to_csv",
    "write_tracking_overlay_tiff",
]
