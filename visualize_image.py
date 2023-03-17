import os

import numpy as np
from tqdm import tqdm
from utils.model.generate_image import generate_image
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from visualization.display_image import display_image
import nibabel as nib

numpy_image = True
image_path = 'data/npy/concatenated/CHUM-006.npy'

image = nib.load(image_path).get_fdata() if not numpy_image else np.load(image_path)


display_image(image)

