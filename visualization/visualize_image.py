import os

from matplotlib import pyplot as plt


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


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import SimpleITK as sitk

import numpy as np
from tqdm import tqdm
from utils.inference.generate_image import generate_image
from visualization.display_image import display_image
import nibabel as nib

numpy_image = False
image_path = 'results/cropped_non_normalized_interpolated/generated_images/generated_images/nifti/image_1.nii.gz'

image = nib.load(image_path).get_fdata() if not numpy_image else np.load(image_path)

display_image(image)
plot_center_sagittal_slice('synthetic ', image)
