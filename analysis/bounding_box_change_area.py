"""
Quick plotting template: absolute bounding-box area change over time.

Before running this script, define:
- `data`: a pandas DataFrame with at least `tracked_spindle_number`,
  `frame_number`, and `bounding_box_area` columns.
- `colors_combined`: a color list/array used for per-object line colors.

This file is intentionally lightweight and geared for ad-hoc analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load your data here.
# data = pd.read_csv('path_to_your_csv_file.csv')

plt.figure(figsize=(14, 8))

# Fallback colormap reference (kept for interactive customization).
cm = plt.get_cmap('tab20')  # This colormap provides 20 distinct colors

# Group by tracked spindle ID and plot each object trajectory.
for i, (key, grp) in enumerate(data.groupby(['tracked_spindle_number'])):
    plt.plot(grp['frame_number'], grp['bounding_box_area'], label=f'Object {key}', color=colors_combined[i % 40])

plt.title('Bounding Box Size Change Over Time')
plt.xlabel('Time Frame')
plt.ylabel('Area of the Bounding Box')
plt.legend()
plt.grid(True)

# Save the figure to a JPEG file.
plt.savefig('/mnt/data/bounding_box_size_change.jpg', format='jpg', dpi=300)

plt.show()
