import glob
import os

from scipy import ndimage
import numpy as np

import nibabel as nib

from visualization.display_image import display_image


def connected_component_analysis(image, threshold=0.5, size_threshold=200):
    binary_mask = image > threshold

    # Label the connected components in the binary mask
    labeled_mask, num_labels = ndimage.label(binary_mask)

    # Compute the size of each connected component
    component_sizes = ndimage.sum(binary_mask, labeled_mask, range(1, num_labels + 1))

    # Keep only the connected components that meet a certain size threshold
    filtered_mask = np.zeros_like(labeled_mask)
    for i in range(1, num_labels + 1):
        if component_sizes[i - 1] >= size_threshold:
            filtered_mask[labeled_mask == i] = 1

    # Apply the filtered mask to the original image to remove noise
    return image * filtered_mask

experiment = 'normalized15mm'
i = 0
samples = 524
if not os.path.exists(f'../data/combined/{experiment}'):
    os.makedirs(f'../data/combined/{experiment}')

if not os.path.exists(f'../data/combined/{experiment}/images'):
    os.makedirs(f'../data/combined/{experiment}/images')

if not os.path.exists(f'../data/combined/{experiment}/masks'):
    os.makedirs(f'../data/combined/{experiment}/masks')


for x in glob.glob(f'../data/generated_images/{experiment}/nifti/*.nii.gz'):
    im1 = nib.load(x)
    im1_data = im1.get_fdata()

    image = im1_data[:, :, :64]
    mask = im1_data[:, :, 64:]

    mask = connected_component_analysis(mask)

    if np.unique(mask)[0] == 0 and len(np.unique(mask)) == 1:
        # Discount images that don't have any masks
        continue

    i += 1
    nifti_image = nib.Nifti1Image(image, affine=np.eye(4))
    nifti_mask = nib.Nifti1Image(mask, affine=np.eye(4))

    nib.save(nifti_image, f'../data/combined/{experiment}/images/gen_image_{i}.nii.gz')
    nib.save(nifti_mask, f'../data/combined/{experiment}/masks/gen_mask_{i}.nii.gz')

    if i >= samples:
        break
