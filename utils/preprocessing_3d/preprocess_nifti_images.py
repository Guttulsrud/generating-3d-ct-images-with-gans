import glob
import numpy as np
import matplotlib as mpl
import nibabel as nib
import os
from tqdm import tqdm

from visualization.display_image import display_image

mpl.use('TkAgg')


def pad_nifti(nifti_image, desired_size, dimensions):
    # Get the current size of the specified dimensions
    current_size = [nifti_image.shape[i] for i in range(3) if i in dimensions]

    # Determine the amount of padding needed to reach the desired size
    pad_size = [max(0, desired_size - size) for size in current_size]

    # Pad the image with zeros using numpy.pad
    pad_width = [(0, 0) if i not in dimensions else (0, pad_size[dimensions.index(i)]) for i in range(3)]
    new_image = np.pad(nifti_image.get_fdata(), pad_width, mode='constant', constant_values=0)

    # Update the affine transformation matrix to reflect the new image dimensions
    new_affine = np.copy(nifti_image.affine)
    new_shape = list(nifti_image.shape)
    for i, dim in enumerate(dimensions):
        new_shape[dim] += pad_size[i]
    new_affine[:3, :3] *= np.diag(
        [new_size / old_size for old_size, new_size in zip(nifti_image.shape[:3], new_shape[:3])])

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


def correct_affine_matrix(new_image, axis=1):
    # Update the affine matrix for new_image
    old_shape = new_image.shape
    if axis == 1:
        new_image = new_image[:, :-1, :]
    elif axis == 2:
        new_image = new_image[:, :, :-1]
    else:
        raise ValueError('axis must be 1 or 2')
    new_shape = new_image.shape
    new_image_affine[1, 1] = new_image_affine[1, 1] * old_shape[1] / new_shape[1]
    return new_image, new_image_affine


if not os.path.exists('../../data/3d/preprocessed/images'):
    os.mkdir('../../data/3d/preprocessed/images')

if not os.path.exists('../../data/3d/preprocessed/masks'):
    os.mkdir('../../data/3d/preprocessed/masks')

if not os.path.exists('../../data/3d/preprocessed/concatenated'):
    os.mkdir('../../data/3d/preprocessed/concatenated')

path = '../../data/original_size/training'
image_paths = glob.glob(os.path.join(f'{path}/images', '*CT.nii.gz'))
label_paths = glob.glob(os.path.join(f'{path}/labels', '*.nii.gz'))
i = 0

invalid_images = 0
valid_images = 0
for image_path, mask_path in tqdm(zip(image_paths, label_paths)):

    if i == 1:
        exit()

    i+=1

    image = nib.load(image_path)
    mask = nib.load(mask_path)

    if image.shape[0] != 512 or image.shape[1] != 512 or image.shape != mask.shape:
        invalid_images += 1
        continue

    # # Mask is bigger than image for some reason, chop mask
    # if image.shape[0] == mask.shape[0] - 1:
    #     mask_data = mask.get_fdata()
    #     mask = nib.Nifti1Image(mask_data[:-1, :, :], np.copy(image.affine), mask.header)
    #
    # if image.shape[1] == mask.shape[1] - 1:
    #     mask_data = mask.get_fdata()
    #     mask = nib.Nifti1Image(mask_data[:, :-1, :], np.copy(image.affine), mask.header)
    #
    # if image.shape[2] == mask.shape[2] - 1:
    #     mask_data = mask.get_fdata()
    #     mask = nib.Nifti1Image(mask_data[:, :, :-1], np.copy(image.affine), mask.header)

    # Image and mask must have 256 in the last dimension, in this case the image is too big, chop center 256 slices
    if image.shape[2] > 256:
        new_image, new_image_affine = chop_nifti(image)
        new_mask, new_mask_affine = chop_nifti(mask)
    else:
        # Last dimension is less than 256, pad
        new_image, new_image_affine = pad_nifti(image, desired_size=256, dimensions=[2])
        new_mask, new_mask_affine = pad_nifti(mask, desired_size=256, dimensions=[2])

    mask = nib.Nifti1Image(new_mask, new_mask_affine)
    image = nib.Nifti1Image(new_image, new_image_affine)

    #
    # if new_image.shape[1] > 512:
    #     # Update the affine matrix for new_image
    #     new_image, new_imagine_affine = correct_affine_matrix(new_image)
    #
    #     # Use the updated affine matrix for new_mask
    #     new_mask_affine = np.copy(new_image_affine)
    #
    # if new_mask.shape[1] > 512:
    #     # Update the affine matrix for new_mask
    #     new_mask, new_mask_affine = correct_affine_matrix(new_mask)
    #
    #     # Use the updated affine matrix for new_mask
    #     new_image_affine = np.copy(new_mask_affine)

    # new_mask_affine = np.copy(new_image_affine)
    # Check if the affine matrices match
    if not np.allclose(new_image_affine, new_mask_affine):
        raise ValueError("Affine matrices for input images do not match")

    image_path = os.path.join("../../data/3d/preprocessed/images", image_path.split('images\\')[-1])
    mask_path = os.path.join("../../data/3d/preprocessed/masks", mask_path.split('labels\\')[-1])
    nib.save(image, image_path)
    nib.save(mask, mask_path)

    img1 = nib.load(image_path)
    img2 = nib.load(mask_path)

    path = image_path.split('images\\')[-1]
    path = path.replace('__CT', '')

    concat_img = nib.concat_images([img1, img2], axis=2)
    nib.save(concat_img, f'../../data/3d/preprocessed/concatenated/{path}')
    img = nib.load(f'../../data/3d/preprocessed/concatenated/{path}')
    # display_image(img.get_fdata())
    valid_images += 1
    display_image(img1.get_fdata())
    display_image(img2.get_fdata())
    display_image(img.get_fdata())

print(f'Invalid images: {invalid_images}')
print(f'Valid images: {valid_images}')
