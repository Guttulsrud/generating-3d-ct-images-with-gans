import glob
import os

from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import SimpleITK as sitk

import numpy as np
from tqdm import tqdm
from utils.model.generate_image import generate_image
from visualization.display_image import display_image
import nibabel as nib
def plot_center_sagittal_slice(data, name):
    data = np.transpose(data, axes=(1, 0, 2))
    center_sagittal_index = data.shape[1] // 2
    center_sagittal_slice = data[:, center_sagittal_index, :]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(np.rot90(center_sagittal_slice), cmap='gray', aspect='equal',
                   extent=[0, data.shape[0], 0, data.shape[0]])
    ax.set_title(f'{name} Center Sagittal')
    ax.axis('off')
    plt.show()
numpy_image = True
image_path = 'data/npy/cropped_non_normalized_interpolated/CHUV-031.npy'

for image_path in glob.glob('data/npy/cropped_non_normalized_interpolated/*.npy'):
    image = nib.load(image_path).get_fdata() if not numpy_image else np.load(image_path)
    plot_center_sagittal_slice(image, image_path)
    display_image(image)

