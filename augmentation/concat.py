import glob
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

data_dir = '../data/original'
output_dir = '../data/concatenated'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

train_images = sorted(
    glob.glob(os.path.join(data_dir, "images", "*CT.nii.gz")))
train_labels = sorted(
    glob.glob(os.path.join(data_dir, "masks", "*.nii.gz")))

wierd_shapes = 0


for image, mask in tqdm(zip(train_images, train_labels)):
    # nifti_img1 = nib.load(image)
    # data1 = nifti_img1.get_fdata()
    # affine1 = nifti_img1.affine
    #
    # nifti_img2 = nib.load(mask)
    # data2 = nifti_img2.get_fdata()
    # affine2 = nifti_img2.affine
    #
    # # Check if the affine matrices are compatible
    # if not np.allclose(affine1, affine2):
    #     print('The affine matrices of the two images are not compatible.')
    #     exit()
    #
    # continue
    # # Get the number of slices in the images
    # num_slices = data1.shape[2]
    #
    # # Concatenate the two images slice by slice
    # concatenated_data = np.empty((data1.shape[0], data1.shape[1], num_slices * 2))
    #
    # bad_image = False
    # for i in range(num_slices):
    #     concatenated_data[:, :, i * 2] = data1[:, :, i]
    #     try:
    #         concatenated_data[:, :, i * 2 + 1] = data2[:, :, i]
    #     except Exception as e:
    #         bad_image = True
    #
    # if bad_image:
    #     wierd_shapes += 1
    #     print(f'Image: {image}, shape: {data1.shape}')
    #     continue
    # # Update the affine matrix to reflect the concatenation
    # new_affine = affine1.copy()
    # new_affine[:3, 3] += affine1[:3, 2] * num_slices
    nifti_img1 = nib.load(image)
    data1 = nifti_img1.get_fdata()
    affine1 = nifti_img1.affine

    nifti_img2 = nib.load(mask)
    data2 = nifti_img2.get_fdata()
    affine2 = nifti_img2.affine

    # Get the shape of each image
    shape1 = data1.shape
    shape2 = data2.shape

    # Find the maximum shape in the second dimension between the two datasets
    max_shape = max(shape1[1], shape2[1])

    # Create a new array for concatenated_data with the larger shape
    concatenated_data = np.empty((shape1[0], max_shape, shape1[2] + shape2[2]))

    # Assign the slices from data1 and data2 to concatenated_data using indexing
    concatenated_data[:, :shape1[1], :shape1[2]] = data1
    concatenated_data[:, :shape2[1], shape1[2]:] = data2

    # Update the affine matrix to reflect the concatenation
    new_affine = affine1.copy()
    new_affine[:2, 2] *= max_shape / shape1[1]
    new_affine[:3, 3] += affine1[:3, 2] * shape1[2]

    # Save the concatenated image to a new Nifti file
    new_nifti_img = nib.Nifti1Image(concatenated_data, new_affine)

    concat_path = mask.split('\masks\\')[-1].replace('__CT', '')
    nib.save(new_nifti_img, f'{output_dir}/{concat_path}')

# broken images:
# CHUP-008__CT,
# CHUP-020__CT,
# CHUP-036__CT
# CHUP-068__CT
# CHUS-009__CT
# CHUS-086__CT
# MDA-031__CT
# MDA-108__CT
# MDA-125__CT
