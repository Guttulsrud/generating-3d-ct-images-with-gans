import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt

from visualization.display_image import display_image


# image_path = 'results/cropped_non_normalized_interpolated/generated_images/generated_images/nifti/image_1.nii.gz'
#
# nifti_img = nib.load(image_path)
#
# # Get the image data as a numpy array
# img_data = nifti_img.get_fdata()
#
# # Split the image into two arrays, each with shape 128x128x64
# first_half = img_data[:, :, :64]
# second_half = img_data[:, :, 64:]
#
# # Create new NIfTI images from the split arrays
# first_half_nifti = nib.Nifti1Image(first_half, nifti_img.affine)
# second_half_nifti = nib.Nifti1Image(second_half, nifti_img.affine)
#
# # Save the new NIfTI images to disk
# nib.save(first_half_nifti, 'first_half.nii')
# nib.save(second_half_nifti, 'second_half.nii')

def plot_center_sagittal_slice_with_mask(image, mask):
    data = np.transpose(image, axes=(1, 0, 2))

    # Find sagittal slice with maximum number of nonzero values in the mask
    mask_data = np.transpose(mask, axes=(1, 0, 2))
    mask_sums = np.sum(mask_data, axis=(0, 2))
    center_sagittal_index = 32

    center_sagittal_slice = data[:, center_sagittal_index, :]

    # Find center sagittal slice of the mask
    mask_center_sagittal_slice = mask_data[:, center_sagittal_index, :]

    # Create overlay array with non-zero values only
    overlay = np.zeros_like(center_sagittal_slice)
    overlay[mask_center_sagittal_slice != 0] = mask_center_sagittal_slice[mask_center_sagittal_slice != 0]

    # Overlay the overlay array on top of the center sagittal slice of the image
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(np.rot90(center_sagittal_slice), cmap='gray', aspect='equal',
                   extent=[0, data.shape[0], 0, data.shape[0]])
    ax.imshow(np.rot90(overlay), cmap='Reds', alpha=0.2, aspect='equal', extent=[0, data.shape[0], 0, data.shape[0]])
    ax.set_title(f'Sagittal Slice {center_sagittal_index}')
    ax.axis('off')
    plt.show()

def plot_center_sagittal_slice(name, data):
    data = np.transpose(data, axes=(1, 0, 2))
    center_sagittal_index = data.shape[1] // 2
    center_sagittal_slice = data[:, center_sagittal_index, :]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(np.rot90(center_sagittal_slice), cmap='gray', aspect='equal',
                   extent=[0, data.shape[0], 0, data.shape[0]])
    ax.set_title(f'{name}')
    ax.axis('off')
    plt.show()

image = nib.load('first_half.nii').get_fdata()
mask = nib.load('second_half.nii').get_fdata()
plot_center_sagittal_slice_with_mask(image, mask)