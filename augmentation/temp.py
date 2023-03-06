import numpy as np
import nibabel as nib
from monai.transforms import RandFlip
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# load your 3D image as a numpy array
path = '../data/3d/preprocessed/concatenated/CHUM-001.nii.gz'
image = nib.load(path)
affine = image.affine
data = image.get_fdata()

# create the RandFlip transform with probability=0.5 for each axis
flip_transform = RandFlip(prob=0.5, spatial_axis=(0, 1, 2))

# apply the transform to the image
augmented_image = flip_transform(image)

# save the augmented image as a new .nii file
nib.save(nib.Nifti1Image(augmented_image, affine=affine), '../data/3d/CHUM-001.nii.gz')
