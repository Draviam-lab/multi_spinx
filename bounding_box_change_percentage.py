import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file to check its content
file_path = '/mnt/data/tracked_spindles_summary_frame_1_to_25.csv'
data = pd.read_csv(file_path)

# Calculate the area of the bounding boxes
data['area'] = (data['max_row'] - data['min_row']) * (data['max_col'] - data['min_col'])

# Normalize the areas: the initial area of each bounding box is set to 1
normalized_areas = data.groupby('tracked_spindle_number')['area'].transform(lambda x: x / x.iloc[0])

# Add the normalized area to the dataframe
data['normalized_area'] = normalized_areas

# Plotting adjustments: different colors for each object and exporting the plot
plt.figure(figsize=(10, 6))

# Generate a color map to ensure different objects have different colors
color_map = plt.cm.get_cmap('hsv', len(data['tracked_spindle_number'].unique()) + 1)

for idx, obj_id in enumerate(data['tracked_spindle_number'].unique(), start=0):
    subset = data[data['tracked_spindle_number'] == obj_id]
    plt.plot(subset['frame_number'], subset['normalized_area'], label=f'Object {obj_id}', color=color_map(idx))

plt.xlabel('Frame Number')
plt.ylabel('Area of the Bounding Box (Normalized)')
plt.title('Movie 1')
plt.legend()
plt.grid(True)

# Exporting the plot
export_path = '/mnt/data/bounding_box_change_percentage_movie1.jpg'
plt.savefig(export_path)
