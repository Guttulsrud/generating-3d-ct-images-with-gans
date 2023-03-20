import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import nibabel as nib
import seaborn as sns

image_paths = glob.glob('../data/original/images/*CT.nii.gz')
mask_paths = glob.glob('../data/original/masks/*.nii.gz')
data = []

lens = []

for image_path, mask_path in tqdm(zip(image_paths, mask_paths)):
    image = nib.load(image_path)
    # if image.shape[2] < 300:
    #     continue

    group = image_path.split('images\\')[1].split('-')[0]
    data.append({'source': group, 'slices': image.shape[2]})

data = pd.DataFrame(data)

# # Set the figure size and style
# plt.figure(figsize=(10, 6))
# # plt.style.use('seaborn-whitegrid')
#
# # Create the histogram
# for group in data['source'].unique():
#     plt.hist(data[data['source'] == group]['slices'], bins=20, alpha=0.5, label=group)
#
# # Set the title, labels and legend
# plt.title('Slice distribution per datasource')
# plt.xlabel('Number of slices')
# plt.ylabel('Frequency')
#
# # Adjust the legend fontsize
# legend = plt.legend(title='Datasource')
# plt.setp(legend.get_title())
#
# plt.savefig('slice_distribution.png')
# plt.show()


unique_groups = data['source'].unique()
n_groups = len(unique_groups)

# Set the figure size and style
# plt.style.use('seaborn-whitegrid')

# Create a grid of subplots with 2 columns
n_columns = 4
n_rows = int(np.ceil(n_groups / n_columns))
fig, axes = plt.subplots(n_rows, n_columns, figsize=(12, 4 * n_rows))

for i, group in enumerate(unique_groups):
    # Get the current subplot's row and column index
    row_idx = i // n_columns
    col_idx = i % n_columns

    # Plot the histogram for the current group
    axes[row_idx, col_idx].hist(data[data['source'] == group]['slices'], bins=15)

    # Set the title for the current subplot
    axes[row_idx, col_idx].set_title(group)

    # Set the x and y labels for the current subplot
    axes[row_idx, col_idx].set_xlabel('Slices')
    axes[row_idx, col_idx].set_ylabel('Frequency')

# Remove empty subplots if any
if n_groups % n_columns != 0:
    for j in range(n_groups, n_rows * n_columns):
        fig.delaxes(axes[j // n_columns, j % n_columns])

plt.subplots_adjust(hspace=0.4, wspace=0.4)

# Add a main title for the entire plot
plt.suptitle('Slice distribution per datasource', fontsize=16)
plt.savefig('slice_distribution_separate.png')
plt.show()
