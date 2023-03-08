import math

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt

path = f'../data/3d/preprocessed/concatenated/CHUM-001.nii.gz'

img = nib.load(path)
data = img.get_fdata()

nslices = data.shape[2]

ncols = int(math.ceil(math.sqrt(nslices)))
nrows = int(math.ceil(nslices/float(ncols)))

# Calculate the number of slices to display
nslices = data.shape[2]

# Create a figure with the appropriate size and layout
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))

# Loop through the slices and display each one on a subplot
for i, ax in enumerate(axes.flat):
    if i < nslices:
        ax.imshow(data[:, :, i], cmap='viridis')
        ax.axis('off')
    else:
        ax.set_visible(False)

# Adjust the spacing between the subplots
fig.subplots_adjust(wspace=0.1, hspace=0.1)

plt.tight_layout()
# Show the figure
plt.show()