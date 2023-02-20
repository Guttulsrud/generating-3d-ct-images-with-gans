import glob
import os
from scipy.ndimage import zoom
import matplotlib as mpl

path = f'../data'
mpl.use('TkAgg')  # !IMPORTANT

import numpy as np
import tensorflow as tf
import nibabel as nib
import glob
import os
from config import config
from tqdm import tqdm
import seaborn as sns
import ast
import matplotlib.pyplot as plt


def pad_image(image, shape):
    return np.pad(image, [(0, shape[i] - image.shape[i]) for i in range(3)], 'constant', constant_values=0)


image_paths = glob.glob(os.path.join(f'{path}/resampled/resampled_0_3/images', '*CT.nii.gz'))
label_paths = glob.glob(os.path.join(f'{path}/resampled/resampled_0_3/labels', '*.nii.gz'))
max_x_image = 0
max_y_image = 0
max_z_image = 0
max_x_label = 0
max_y_label = 0
max_z_label = 0





for image_path, label_path in tqdm(zip(image_paths, label_paths)):
    image = nib.load(image_path)
    label = nib.load(label_path)

    image_data = image.get_fdata()
    label_data = label.get_fdata()

    # image_data = zoom(image_data, (0.3, 0.3, 0.3))  # Resample the image to smaller shape
    # label_data = zoom(label_data, (0.3, 0.3, 0.3))  # Resample the image to smaller shape

    # resampled_image = zoom(image, (0.3, 0.3, 0.3))  # Resample the image to smaller shape
    # resampled_label = zoom(label, (0.3, 0.3, 0.3))
    # concat = np.concatenate([resampled_image, resampled_label], 2)

    # if image_data.shape[0] > max_x_image:
    #     max_x_image = image_data.shape[0]
    #
    # if image_data.shape[1] > max_y_image:
    #     max_y_image = image_data.shape[1]
    #
    # if image_data.shape[2] > max_z_image:
    #     max_z_image = image_data.shape[2]
    #
    # if label_data.shape[0] > max_x_label:
    #     max_x_label = label_data.shape[0]
    #
    # if label_data.shape[1] > max_y_label:
    #     max_y_label = label_data.shape[1]
    #
    # if label_data.shape[2] > max_z_label:
    #     max_z_label = label_data.shape[2]
    #
    # with open(f'images.txt', "a") as log:
    #     log.write(f'{[x for x in image_data.shape]}\n')
    #
    # with open(f'labels.txt', "a") as log:
    #     log.write(f'{[x for x in image_data.shape]}\n')

    # padded_image = pad_image(image_data, (104, 104, 206))  # Pad images so they have the size of the biggest image
    # padded_label = pad_image(label_data, (104, 104, 206))
    #
    # resampled_label = zoom(label_data, (0.5, 0.5, 0.5))  # Resample the image to smaller shape
    #
    # #
    x = nib.Nifti1Image(image_data, image.affine)
    y = nib.Nifti1Image(label_data, label.affine)
    #
    nib.save(x, os.path.join("../data/resampled/resampled_0_3/chopped/images", image_path.split('images\\')[-1]))
    nib.save(y, os.path.join("../data/resampled/resampled_0_3/chopped/labels", label_path.split('labels\\')[-1]))

# print(f'image: {max_x_image}, {max_y_image}, {max_z_image}')
# print(f'label: {max_x_label}, {max_y_label}, {max_z_label}')
