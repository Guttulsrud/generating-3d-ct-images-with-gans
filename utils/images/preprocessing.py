import glob
import os

import numpy as np
from scipy.ndimage import zoom
import matplotlib as mpl
import nibabel as nib
import os
from tqdm import tqdm

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


def chop_pad_images():
    dataset = '0.075'

    image_paths = glob.glob(os.path.join(f'{path}/chopped/{dataset}/images', '*CT.nii.gz'))
    label_paths = glob.glob(os.path.join(f'{path}/chopped/{dataset}/labels', '*.nii.gz'))

    for image_path, label_path in tqdm(zip(image_paths, label_paths)):
        image = nib.load(image_path)
        label = nib.load(label_path)
        image_data = image.get_fdata()
        label_data = label.get_fdata()
        print(image.shape)
        continue

        if dataset == 'resampled_0_15':
            z_dimension = 39
        elif dataset == 'resampled_0_075':
            z_dimension = 19
        else:
            z_dimension = 27

        image_shape = (image_data.shape[0], image_data.shape[1], z_dimension)

        if image_data.shape[2] > z_dimension:
            image_data = chop_image(image_data, z_dimension)
            label_data = chop_image(label_data, z_dimension)
        else:
            image_data = pad_image(image_data, image_shape)
            label_data = pad_image(label_data, image_shape)

        if dataset == 'resampled_0_15':
            image_data = pad_image(image_data, (78, 78, 39))
            label_data = pad_image(label_data, (78, 78, 39))

        if image_data.shape != label_data.shape:
            print('BAD', ' ', image_path)
        if image_data.shape != (38, 38, 19):
            continue
        x = nib.Nifti1Image(image_data, image.affine)
        y = nib.Nifti1Image(label_data, label.affine)

        nib.save(x, os.path.join("../../data/chopped/0.075/images", image_path.split('images\\')[-1]))
        nib.save(y, os.path.join("../../data/chopped/0.075/labels", label_path.split('labels\\')[-1]))


chop_pad_images()
