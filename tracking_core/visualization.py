"""Visualization and overlay export helpers for tracking outputs."""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imsave


def bounding_box_plot(img, bbox_list, linewidth):
    """
    Display a single 2D frame with red bounding-box overlays.

    Parameters
    ----------
    img : ndarray
        Source image to display.
    bbox_list : list[tuple]
        Bounding boxes in `(min_row, min_col, max_row, max_col)` format.
    linewidth : int
        Rectangle line width.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    for bboxes in bbox_list:
        minr, minc, maxr, maxc = bboxes
        rect = mpatches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=linewidth,
        )
        ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def _max_projection(img_5d, time_index, channel, channel_axis_last):
    """Internal helper to max-project one channel for one frame."""
    if channel_axis_last:
        return np.max(img_5d[time_index, :, :, :, channel], axis=0)
    return np.max(img_5d[time_index, :, channel, :, :], axis=0)


def write_tracking_overlay_tiff(
    img_path,
    output_path,
    nr_frame,
    channel,
    start_frame,
    tracked_items,
    tracked_id_key,
    channel_axis_last,
):
    """
    Render per-frame tracking overlays and write them as a TIFF stack.

    Parameters
    ----------
    img_path : str
        Path to source 5D TIFF.
    output_path : str
        Destination path for overlay TIFF stack.
    nr_frame : int
        Number of frames to render.
    channel : int
        Channel index used for background visualization.
    start_frame : int
        Zero-based start frame.
    tracked_items : list[dict]
        Records containing `frame_number`, `bounding_box`, and tracking id.
    tracked_id_key : str
        Dictionary key for the displayed track identifier.
    channel_axis_last : bool
        True for `{T, Z, X, Y, C}`, False for `{T, Z, C, X, Y}`.
    """
    img_5d = imread(img_path)
    num_time_points = nr_frame
    output_images = []

    for t in range(num_time_points):
        max_projected_img = _max_projection(
            img_5d, t + start_frame, channel, channel_axis_last
        )

        width_px, height_px = np.shape(max_projected_img)
        dpi = 100
        width_in = width_px / dpi
        height_in = height_px / dpi

        fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
        ax.imshow(max_projected_img, cmap="gray")

        tracked_items_at_frame = [
            item for item in tracked_items if item.get("frame_number") == (t + start_frame)
        ]

        for i in range(len(tracked_items_at_frame)):
            minr, minc, maxr, maxc = tracked_items_at_frame[i]["bounding_box"]
            rect = mpatches.Rectangle(
                (minc, minr),
                maxc - minc,
                maxr - minr,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
            ax.add_patch(rect)

            tracked_id = tracked_items_at_frame[i][tracked_id_key]
            if tracked_id is not None:
                ax.text(minc + 5, minr + 25, str(tracked_id), color="red", fontsize=18)
            elif tracked_id is None:
                ax.text(minc + 5, minr + 25, "new", color="red", fontsize=18)

        ax.set_axis_off()
        plt.subplots_adjust(
            left=0, right=1, bottom=0, top=1, wspace=0, hspace=0
        )
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())
        output_images.append(data)

        plt.close(fig)

    imsave(output_path, np.array(output_images))
