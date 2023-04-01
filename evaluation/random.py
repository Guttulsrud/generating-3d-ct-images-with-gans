import numpy as np
import nibabel as nib

for x in range(1, 600):
    # Create 3D array with random values
    arr = np.random.rand(128, 128, 128)

    # Save as NIfTI image
    img = nib.Nifti1Image(arr, np.eye(4))  # create NIfTI image
    nib.save(img, f'data/noise_{x}.nii.gz')  # save to file
