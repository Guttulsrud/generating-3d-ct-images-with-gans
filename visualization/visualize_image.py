import os

from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import SimpleITK as sitk

import numpy as np
from tqdm import tqdm
from utils.inference.generate_image import generate_image
from visualization.display_image import display_image
import nibabel as nib

numpy_image = False
image_path = '../data/generated_images/256_normalized15mm/nifti/raw/image_30.nii.gz'

image = nib.load(image_path).get_fdata() if not numpy_image else np.load(image_path)

fig = display_image(image, return_figure=True, show=True)

# plt.savefig('hounsfield_gen.png', dpi=300, bbox_inches='tight')
