import numpy as np

from visualization.display_image import display_image

img = np.load('../data/complete_dataset_ppy/CHUM-001_normalized.nii.gz.npy')

display_image(img)