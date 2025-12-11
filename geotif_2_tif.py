import numpy as np
from osgeo import gdal
from PIL import Image

input_path = r'D:\OneDrive - University of Calgary\Desktop\change_detection_results\TED\landcover_2010_prediction.tif'
output_path = r'D:\OneDrive - University of Calgary\Desktop\change_detection_results\TED\landcover_2010_prediction_palette.tif'

# List of hex colors
# hex_colors = [
#     '#fffaff', '#003d00', '#949c70', '#006300', '#1eab05', '#148c3d',
#     '#5c752b', '#b39e2b', '#b38a33', '#e8dc5e', '#e1cf8a', '#9c7554',
#     '#bad48f', '#408a70', '#6ba38a', '#e6ae66', '#a8abae', '#dc2126',
#     '#4c70a3', '#fffaff'
# ]

hex_colors = [
    '#fffaff', '#003d00', '#949c70',  '#1eab05',
    '#5c752b', '#b38a33', '#e1cf8a', '#e6ae66', '#a8abae', '#dc2126',
    '#4c70a3', '#fffaff'
]

# Convert hex to RGB
def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

palette_rgb = [hex_to_rgb(h) for h in hex_colors]

# Flatten and pad palette
palette = []
for rgb in palette_rgb:
    palette.extend(rgb)
palette += [0, 0, 0] * (256 - len(palette_rgb))

ds = gdal.Open(input_path)
arr = ds.ReadAsArray().astype(np.uint8)

# Remap arr values to 0..len(palette_rgb)-1, others to 0
arr_idx = np.where((arr >= 0) & (arr < len(palette_rgb)), arr, 0)

img = Image.fromarray(arr_idx.astype(np.uint8), mode='P')
img.putpalette(palette)
img = img.convert('RGB')
img.save(output_path)
print(f"Saved: {output_path}")