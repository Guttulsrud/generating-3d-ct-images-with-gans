import os

import numpy as np
from tqdm import tqdm
from utils.model.generate_image import generate_image
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from visualization.display_image import display_image
import nibabel as nib

numpy_image = False
image_path = 'data/concatenated/MDA-052.nii.gz'

image = nib.load(image_path).get_fdata() if not numpy_image else np.load(image_path)


display_image(image)

