import nibabel as nib
import glob
import os
import scipy as sp
from tqdm import tqdm
import scipy.ndimage as ndimage


def resample(percent):
    images = glob.glob(os.path.join(f'../data/unzipped/training/images', '*CT.nii.gz'))
    labels = glob.glob(os.path.join(f'../data/unzipped/training/labels', '*.nii.gz'))
    for image_path, label_path in tqdm(zip(images, labels)):
        image = nib.load(image_path)
        label = nib.load(label_path)
        image_path = image_path.split('\\')[1]
        label_path = label_path.split('\\')[1]

        image_data = image.get_fdata()
        label_data = label.get_fdata()

        resampled_image = ndimage.zoom(image_data, (percent, percent, percent))
        resampled_label = ndimage.zoom(label_data, (percent, percent, percent))

        x = nib.Nifti1Image(resampled_image, image.affine)
        y = nib.Nifti1Image(resampled_label, label.affine)
        #
        nib.save(x, os.path.join("../data/resampled/resampled_0_03/training/images/", image_path))
        nib.save(y, os.path.join("../data/resampled/resampled_0_03/training/labels/", label_path))


resample(percent=0.03)
