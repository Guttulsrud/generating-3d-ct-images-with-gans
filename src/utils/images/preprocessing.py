import glob
import os
from scipy.ndimage import zoom
import matplotlib as mpl
import nibabel as nib
import os
from tqdm import tqdm

path = f'../data'
mpl.use('TkAgg')


def chop_image(image, slices):
    z_start = (image.shape[2] - slices) // 2
    z_end = z_start + slices

    return image[:, :, z_start:z_end]


def chop_images(slices=27):
    image_paths = glob.glob(os.path.join(f'{path}/original_size/training/images', '*CT.nii.gz'))
    label_paths = glob.glob(os.path.join(f'{path}/original_size/training/labels', '*.nii.gz'))

    for image_path, label_path in tqdm(zip(image_paths, label_paths)):
        image = nib.load(image_path)
        label = nib.load(label_path)
        image_data = image.get_fdata()
        label_data = label.get_fdata()
        image_data = chop_image(image_data, slices)
        label_data = chop_image(label_data, slices)

        x = nib.Nifti1Image(image_data, image.affine)
        y = nib.Nifti1Image(label_data, label.affine)
        #
        nib.save(x, os.path.join("../data/resampled/resampled_0_3/chopped/images", image_path.split('images\\')[-1]))
        nib.save(y, os.path.join("../data/resampled/resampled_0_3/chopped/labels", label_path.split('labels\\')[-1]))