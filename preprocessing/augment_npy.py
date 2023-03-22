import glob
from scipy.ndimage import rotate
import cv2

import numpy as np
import nibabel as nib

from visualization.display_image import display_image
from visualize_image import plot_center_sagittal_slice

for image_path in glob.glob('../data/npy/cropped_non_normalized_interpolated/*.npy'):
    image = np.load(image_path)
    image = np.fliplr(image)
    plot_center_sagittal_slice(image, image_path)
    display_image(image)