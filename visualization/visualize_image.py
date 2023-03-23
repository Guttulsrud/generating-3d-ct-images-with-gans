import os

from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import SimpleITK as sitk

import numpy as np
from tqdm import tqdm
from utils.inference.generate_image import generate_image
from visualization.display_image import display_image
import nibabel as nib

numpy_image = True
image_path = '../data/npy/test/CHUM-006__CT.npy'

image = nib.load(image_path).get_fdata() if not numpy_image else np.load(image_path)

display_image(image)
