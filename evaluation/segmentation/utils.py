import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import math

# Load ground truth and prediction mask images
gt_img = nib.load('../../data/128/ground_truth/masks/CHUM-011.nii.gz')
pred_img = nib.load('data/128/thresholded/CHUM-011/CHUM-011_ensemble.nii.gz')

# Get the image data arrays
gt_data = gt_img.get_fdata()
pred_data = pred_img.get_fdata()

# Define colors for ground truth, prediction mask, overlapping values, and non-overlapping values
bg_color = [0, 0, 0]      # black
gt_color = [1, 1, 1]      # white
overlap_color = [0, 1, 0]  # green
nonoverlap_color = [1, 0, 0]  # red

# Create a new image array that combines the ground truth and prediction mask
overlay_data = np.zeros(gt_data.shape + (3,))
overlay_data[gt_data == 0, :] = bg_color
overlay_data[gt_data == 1, :] = gt_color
overlay_data[pred_data == 1, :] += nonoverlap_color
overlay_data[(gt_data == 1) & (pred_data == 1), :] = overlap_color

# Find the indices of slices that contain non-zero values and remove duplicates
nonzero_slices = np.unique(np.where(np.any(overlay_data, axis=(0, 1)))[0])
print("Non-zero slices:", nonzero_slices)

# Set number of columns and rows of subplots based on the number of nonzero slices
n_slices = len(nonzero_slices)
n_cols = 8
n_rows = int(math.ceil(n_slices / float(n_cols)))

# Plot each nonzero slice of the overlay image as a separate subplot
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5))

for i, ax in enumerate(axes.flat):
    if i < n_slices:
        slice_idx = nonzero_slices[i]
        print("Plotting slice", slice_idx)
        ax.imshow(overlay_data[:, :, slice_idx])
        ax.axis('off')
    else:
        ax.set_visible(False)
plt.subplots_adjust(wspace=0.05, hspace=0.01, top=0.9, bottom=0.1, left=0.01, right=0.99)
plt.savefig('CHUM-011_0.87.png', dpi=300)
plt.show()
