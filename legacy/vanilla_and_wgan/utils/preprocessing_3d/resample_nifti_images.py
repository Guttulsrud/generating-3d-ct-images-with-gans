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

        print(resampled_image.shape)
        # x = nib.Nifti1Image(resampled_image, image.affine)
        # y = nib.Nifti1Image(resampled_label, label.affine)
        #
        # # append info to .txt file
        # with open(f'info.txt', 'a') as f:
        #     f.write(
        #         f'{image_path} To {resampled_image.shape} from {image_data.shape}\n')
        # #
        # nib.save(x, os.path.join("../data/resampled/0.5/images/", image_path))
        # nib.save(y, os.path.join("../data/resampled/0.5/labels/", label_path))


resample(percent=0.5)
