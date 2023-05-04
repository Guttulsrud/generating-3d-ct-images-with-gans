import glob
import os

from scipy import ndimage
import numpy as np

import nibabel as nib
from tqdm import tqdm

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


experiment = 'scaled'
i = 0
samples = 524
img_size = 128

hpo = False

configs = [
    {'threshold': 0.1, 'size': 25},
    {'threshold': 0.1, 'size': 50},
    {'threshold': 0.1, 'size': 100},
    {'threshold': 0.1, 'size': 200},

    {'threshold': 0.2, 'size': 25},
    {'threshold': 0.2, 'size': 50},
    {'threshold': 0.2, 'size': 100},
    {'threshold': 0.2, 'size': 200},

    {'threshold': 0.3, 'size': 25},
    {'threshold': 0.3, 'size': 50},
    {'threshold': 0.3, 'size': 100},
    {'threshold': 0.3, 'size': 200},

    {'threshold': 0.4, 'size': 25},
    {'threshold': 0.4, 'size': 50},
    {'threshold': 0.4, 'size': 100},
    {'threshold': 0.4, 'size': 200},

    {'threshold': 0.5, 'size': 25},
    {'threshold': 0.5, 'size': 50},
    {'threshold': 0.5, 'size': 100},
    {'threshold': 0.5, 'size': 200},

    {'threshold': 0.6, 'size': 25},
    {'threshold': 0.6, 'size': 50},
    {'threshold': 0.6, 'size': 100},
    {'threshold': 0.6, 'size': 200},

    {'threshold': 0.7, 'size': 25},
    {'threshold': 0.7, 'size': 50},
    {'threshold': 0.7, 'size': 100},
    {'threshold': 0.7, 'size': 200},

    {'threshold': 0.8, 'size': 25},
    {'threshold': 0.8, 'size': 50},
    {'threshold': 0.8, 'size': 100},
    {'threshold': 0.8, 'size': 200},

]

if not os.path.exists(f'../data/post_processed/{experiment}/cca_masks_hpo'):
    os.makedirs(f'../data/post_processed/{experiment}/cca_masks_hpo')

for config in configs:
    threshold = config['threshold']
    size_threshold = config['size']

    if not os.path.exists(f'../data/post_processed/{experiment}/cca_masks_hpo/{threshold}_{size_threshold}'):
        os.makedirs(f'../data/post_processed/{experiment}/cca_masks_hpo/{threshold}_{size_threshold}')

if not os.path.exists(f'../data/post_processed'):
    os.makedirs(f'../data/post_processed')

if not os.path.exists(f'../data/post_processed/{experiment}'):
    os.makedirs(f'../data/post_processed/{experiment}')

if not os.path.exists(f'../data/post_processed/{experiment}/images'):
    os.makedirs(f'../data/post_processed/{experiment}/images')

if not os.path.exists(f'../data/post_processed/{experiment}/masks'):
    os.makedirs(f'../data/post_processed/{experiment}/masks')

if not os.path.exists(f'../data/post_processed/{experiment}/concat'):
    os.makedirs(f'../data/post_processed/{experiment}/concat')

if not os.path.exists(f'../data/post_processed/{experiment}/cca_masks'):
    os.makedirs(f'../data/post_processed/{experiment}/cca_masks')

if not os.path.exists(f'../data/post_processed/{experiment}/cca_concat'):
    os.makedirs(f'../data/post_processed/{experiment}/cca_concat')

for x in tqdm(glob.glob(f'../data/generated_images/{experiment}/nifti/*.nii.gz')):
    im1 = nib.load(x)
    im1_data = im1.get_fdata()

    image = im1_data[:, :, :img_size // 2]
    mask = im1_data[:, :, img_size // 2:]

    if not hpo:

        cca_mask = connected_component_analysis(mask)

        if np.unique(cca_mask)[0] == 0 and len(np.unique(cca_mask)) == 1:
            # Discount images that don't have any masks
            continue
        cca_concat = np.concatenate((image, cca_mask), axis=2)
        concat = np.concatenate((image, mask), axis=2)

        i += 1
        nifti_image = nib.Nifti1Image(image, affine=np.eye(4))
        nifti_mask = nib.Nifti1Image(mask, affine=np.eye(4))
        nifti_cca_mask = nib.Nifti1Image(cca_mask, affine=np.eye(4))
        nifti_concat = nib.Nifti1Image(concat, affine=np.eye(4))
        nifti_cca_concat = nib.Nifti1Image(cca_concat, affine=np.eye(4))

        nib.save(nifti_image, f'../data/post_processed/{experiment}/images/gen_image_{i}.nii.gz')
        nib.save(nifti_mask, f'../data/post_processed/{experiment}/masks/gen_image_{i}.nii.gz')
        nib.save(nifti_cca_mask, f'../data/post_processed/{experiment}/cca_masks/gen_image_{i}.nii.gz')
        nib.save(nifti_concat, f'../data/post_processed/{experiment}/concat/gen_image_{i}.nii.gz')
        nib.save(nifti_cca_concat, f'../data/post_processed/{experiment}/cca_concat/gen_image_{i}.nii.gz')

        if i >= samples:
            break

    for config in configs:
        threshold = config['threshold']
        size_threshold = config['size']

        cca_mask = connected_component_analysis(mask, threshold=threshold, size_threshold=size_threshold)

        if np.unique(cca_mask)[0] == 0 and len(np.unique(cca_mask)) == 1:
            continue

        nifti_image = nib.Nifti1Image(image, affine=np.eye(4))
        nib.save(nifti_image,
                 f'../data/post_processed/{experiment}/cca_masks_hpo/{threshold}_{size_threshold}/gen_image_{i}.nii.gz')

    i += 1
