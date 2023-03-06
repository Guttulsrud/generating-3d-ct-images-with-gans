import nibabel as nib
import SimpleITK as sitk
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from monai.transforms import (

    RandCropByPosNegLabel,
    RandGaussianNoised,
    Zoomd,
    RandRotated,
    RandAffined,
    Rand3DElastic, Compose,
)
import monai.transforms as transforms

mpl.use('TkAgg')
# flip, rotation, translation, gaussian noise
import os

# Define input and output directories
input_dir = '../data/resampled/resampled_0_3/images'
output_dir = '../data/augmented/3d/'
augmentation_dict = {
    'random_gaussian_noise': RandGaussianNoised(keys='image', prob=0.5, std=0.1),
    'random_zoom': Zoomd(keys='image', prob=0.5, min_zoom=0.9, max_zoom=1.1, mode='nearest', zoom=1.0),
    'random_rotate_90': RandRotated(keys=['image'], prob=0.5, range_x=(-90, 90), range_y=(-90, 90), range_z=(-90, 90),
                                    keep_size=True),
    'random_affine': RandAffined(keys='image', prob=0.5, translate_range=10),
    'random_elastic_deform': Rand3DElastic(prob=0.5, sigma_range=(5, 10), magnitude_range=(0, 5)),
}



def augment(augmentations):
    input_augmentations = []
    for aug in augmentations:
        if aug in augmentation_dict:
            input_augmentations.append(augmentation_dict[aug])
        else:
            raise Exception(f'Augmentation {aug} not found')

    transform = transforms.Compose(input_augmentations)
    print(transform)
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.nii.gz'):
            # Load input image
            path = os.path.join(input_dir, file_name)
            image = nib.load(path)

            continue
            output_image = nib.Nifti1Image(transformed_data, image.affine, image.header)
            nib.save(output_image, os.path.join(output_dir, file_name))


augment(['random_rotate_90'])
