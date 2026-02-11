"""
Quick plotting template: normalized bounding-box area change over time.

This script reads one tracking summary CSV, computes per-object area
normalization using each object's first frame as baseline (=1), and writes a
line plot.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the tracking summary CSV.
file_path = '/mnt/data/tracked_spindles_summary_frame_1_to_25.csv'
data = pd.read_csv(file_path)

# Compute bounding-box area per row.
data['area'] = (data['max_row'] - data['min_row']) * (data['max_col'] - data['min_col'])

# Normalize each spindle's area so its first observation equals 1.
normalized_areas = data.groupby('tracked_spindle_number')['area'].transform(lambda x: x / x.iloc[0])

# Store normalized values for plotting/export if needed.
data['normalized_area'] = normalized_areas

# Build the figure.
plt.figure(figsize=(10, 6))

# Use a deterministic colormap keyed by object index.
color_map = plt.cm.get_cmap('hsv', len(data['tracked_spindle_number'].unique()) + 1)

for idx, obj_id in enumerate(data['tracked_spindle_number'].unique(), start=0):
    subset = data[data['tracked_spindle_number'] == obj_id]
    plt.plot(subset['frame_number'], subset['normalized_area'], label=f'Object {obj_id}', color=color_map(idx))

plt.xlabel('Frame Number')
plt.ylabel('Area of the Bounding Box (Normalized)')
plt.title('Movie 1')
plt.legend()
plt.grid(True)

# Export the plot image.
export_path = '/mnt/data/bounding_box_change_percentage_movie1.jpg'
plt.savefig(export_path)
