import nibabel as nib
import glob
import os
import scipy as sp
from tqdm import tqdm
import scipy.ndimage as ndimage


def down_sample(percent):
    max_x = 0
    max_y = 0
    max_z = 0
    for image_path, label_path in tqdm(
            zip(glob.glob(os.path.join(f'../../data/unzipped/training/images', '*CT.nii.gz')),
                glob.glob(os.path.join(f'../../data/unzipped/training/labels', '*.nii.gz')))):
        image = nib.load(image_path)
        label = nib.load(label_path)
        image_path = image_path.split('\\')[1]
        label_path = label_path.split('\\')[1]

        image_data = image.get_fdata()
        label_data = label.get_fdata()

        # spline interpolation
        # mode: reflect’, ‘grid-mirror’, ‘constant’, ‘grid-constant’, ‘nearest’, ‘mirror’, ‘grid-wrap’, ‘wrap
        down_sampled_image = ndimage.zoom(image_data, (percent, percent, percent))
        down_sampled_label = ndimage.zoom(label_data, (percent, percent, percent))

        if down_sampled_image.shape[0] > max_x:
            max_x = down_sampled_image.shape[0]

        if down_sampled_image.shape[1] > max_y:
            max_y = down_sampled_image.shape[1]

        if down_sampled_image.shape[2] > max_z:
            max_z = down_sampled_image.shape[2]

        if down_sampled_label.shape[0] > max_x:
            max_x = down_sampled_label.shape[0]

        if down_sampled_label.shape[1] > max_y:
            max_y = down_sampled_label.shape[1]

        if down_sampled_label.shape[2] > max_z:
            max_z = down_sampled_label.shape[2]

        x = nib.Nifti1Image(down_sampled_image, image.affine)
        y = nib.Nifti1Image(down_sampled_label, label.affine)
        #
        nib.save(x, os.path.join("../../data/downsampled/50/training/images/", image_path))
        nib.save(y, os.path.join("../../data/downsampled/50/training/labels/", label_path))
        # print(down_sampled_image.shape, down_sampled_label.shape)

    print(max_x, max_y, max_z)
    # 26 26 37 --- 5%


down_sample(percent=0.5)
