"""CSV export helpers for tracked spindle and GFP records."""

import pandas as pd


def _tracks_to_csv(output_path, tracked_items, tracked_id_col, object_number_col):
    """
    Convert tracking dictionaries to the legacy flat CSV format.

    The output schema matches original script behavior:
    tracked id, frame number (1-based), bounding box coordinates, and centroid.
    """
    df = pd.DataFrame(tracked_items)

    df_with_number = df[df[tracked_id_col].notna()]
    df_without_number = df[df[tracked_id_col].isna()]

    df_with_number = df_with_number.sort_values(by=[tracked_id_col, "frame_number"])
    df_without_number = df_without_number.sort_values(
        by=["frame_number", object_number_col]
    )

    sorted_df = pd.concat([df_with_number, df_without_number])

    sorted_df[["min_row", "min_col", "max_row", "max_col"]] = pd.DataFrame(
        sorted_df["bounding_box"].tolist(), index=sorted_df.index
    )
    sorted_df[["centroid_row", "centroid_col"]] = pd.DataFrame(
        sorted_df["centroid"].tolist(), index=sorted_df.index
    )
    sorted_df = sorted_df.drop(columns=["bounding_box", "centroid"])

    sorted_df["frame_number"] = sorted_df["frame_number"] + 1

    sorted_df.to_csv(
        output_path,
        columns=[
            tracked_id_col,
            "frame_number",
            "min_row",
            "min_col",
            "max_row",
            "max_col",
            "centroid_row",
            "centroid_col",
        ],
        index=False,
    )


def spindles_to_csv(output_path, tracked_spindles):
    """Write tracked spindle records to CSV."""
    _tracks_to_csv(
        output_path=output_path,
        tracked_items=tracked_spindles,
        tracked_id_col="tracked_spindle_number",
        object_number_col="spindle_number",
    )


def gfps_to_csv(output_path, tracked_gfps):
    """Write tracked GFP records to CSV."""
    _tracks_to_csv(
        output_path=output_path,
        tracked_items=tracked_gfps,
        tracked_id_col="tracked_gfp_number",
        object_number_col="gfp_number",
    )
