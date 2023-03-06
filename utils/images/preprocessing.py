import glob
import os

import numpy as np
from scipy.ndimage import zoom
import matplotlib as mpl
import nibabel as nib
import os
from tqdm import tqdm
from nibabel.processing import resample_to_output

path = f'../../data'
mpl.use('TkAgg')


def chop_image(image, slices):
    z_start = (image.shape[2] - slices) // 2
    z_end = z_start + slices

    return image[:, :, z_start:z_end]


def pad_image(image, shape):
    pad_width = [(0, shape[i] - image.shape[i]) for i in range(3)]
    padded_image = np.pad(image, pad_width, mode='constant')
    return padded_image


def pad_nifti(nifti_image):
    # Get the current size of the last dimension
    current_size = nifti_image.shape[2]

    # Set the desired size of the last dimension
    desired_size = 256

    # Determine the amount of padding needed to reach the desired size
    pad_size = desired_size - current_size

    # Pad the image with zeros using numpy.pad
    pad_width = ((0, 0), (0, 0), (0, pad_size))
    new_image = np.pad(nifti_image.get_fdata(), pad_width, mode='constant', constant_values=0)

    # Update the affine transformation matrix to reflect the new image dimensions
    new_affine = np.copy(nifti_image.affine)
    new_affine[:3, :3] *= np.diag(
        [new_size / old_size for old_size, new_size in zip(nifti_image.shape[:3], new_image.shape[:3])])

    return new_image, new_affine


def chop_nifti(nifti_image):
    # Get the current size of the last dimension
    current_size = nifti_image.shape[2]

    # Set the desired size of the last dimension
    desired_size = 256

    # Determine the amount of chopping needed to reach the desired size
    chop_size = max(0, current_size - desired_size)

    # Chop the image along the last dimension using numpy indexing
    new_image = nifti_image.get_fdata()[..., :nifti_image.shape[2] - chop_size]

    # Update the affine transformation matrix to reflect the new image dimensions
    new_affine = np.copy(nifti_image.affine)
    new_affine[:3, :3] *= np.diag(
        [new_size / old_size for old_size, new_size in zip(nifti_image.shape[:3], new_image.shape[:3])])

    return new_image, new_affine


def preprocess_nifti():
    dataset = 'original_size'

    image_paths = glob.glob(os.path.join(f'{path}/{dataset}/training/images', '*CT.nii.gz'))
    label_paths = glob.glob(os.path.join(f'{path}/{dataset}/training/labels', '*.nii.gz'))

    for image_path, mask_path in tqdm(zip(image_paths, label_paths)):
        image = nib.load(image_path)
        mask = nib.load(mask_path)

        if image.shape[2] > 256:
            new_image, new_image_affine = chop_nifti(image)
            new_mask, new_mask_affine = chop_nifti(mask)
        else:
            new_image, new_image_affine = pad_nifti(image)
            new_mask, new_mask_affine = pad_nifti(mask)

        image = nib.Nifti1Image(new_image, new_image_affine)
        mask = nib.Nifti1Image(new_mask, new_mask_affine)

        nib.save(image, os.path.join("../../data/3d/preprocessed/images", image_path.split('images\\')[-1]))
        nib.save(mask, os.path.join("../../data/3d/preprocessed/masks", mask_path.split('labels\\')[-1]))


preprocess_nifti()
# image_paths = glob.glob(os.path.join(f'{path}/3d/preprocessed/images', '*CT.nii.gz'))
# mask_paths = glob.glob(os.path.join(f'{path}/3d/preprocessed/masks', '*.nii.gz'))
#
# image_paths = image_paths[:3]
# mask_paths = mask_paths[:3]
# for image_path, mask_path in tqdm(zip(image_paths, mask_paths)):
#     path = os.path.join("../../data/3d/preprocessed/concatenated", image_path.split('images\\')[-1])
#     path = path.replace('__CT', '')
#     img1 = nib.load(image_path)
#     img2 = nib.load(mask_path)
#
#     concat_img = nib.concat_images([img1, img2])
#     nib.save(concat_img, path)
