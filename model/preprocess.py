# resize and rescale images for preprocessing

import glob
import SimpleITK as sitk
import numpy as np
from skimage.transform import resize
import os
import multiprocessing as mp
import nibabel as nib
from visualization.display_image import display_image

### Configs
# 8 cores are used for multi-thread processing
NUM_JOBS = 8
#  resized output size, can be 128 or 256
IMG_SIZE = 128
INPUT_DATA_DIR = '../data/augmented/concat_normalized2.5/images/'
OUTPUT_DATA_DIR = '../data/npy/concatenated/'
# the intensity range is clipped with the two thresholds, this default is used for our CT images, please adapt to your own dataset
LOW_THRESHOLD = -1024
HIGH_THRESHOLD = 600
# suffix (ext.) of input images
SUFFIX = '.nii.gz'
# whether or not to trim blank axial slices, recommend to set as True
TRIM_BLANK_SLICES = False


def resize_img(img):
    nan_mask = np.isnan(img)  # Remove NaN
    img[nan_mask] = LOW_THRESHOLD
    img = np.interp(img, [LOW_THRESHOLD, HIGH_THRESHOLD], [-1, 1])

    if TRIM_BLANK_SLICES:
        valid_plane_i = np.mean(img, (0, 1)) != -1  # Remove blank axial planes
        img = img[:, :, valid_plane_i]

    img = resize(img, (IMG_SIZE, IMG_SIZE, IMG_SIZE), mode='constant', cval=-1)
    return img


def main():
    img_list = list(glob.glob(INPUT_DATA_DIR + "*" + SUFFIX))
    processes = []
    for i in range(NUM_JOBS):
        processes.append(mp.Process(target=batch_resize, args=(i, img_list)))
    for p in processes:
        p.start()


def batch_resize(batch_idx, img_list):
    for idx in range(len(img_list)):
        if idx % NUM_JOBS != batch_idx:
            continue
        imgname = img_list[idx].split('/')[-1]
        if os.path.exists(OUTPUT_DATA_DIR + imgname.split('.')[0] + ".npy"):
            # skip images that already finished pre-processing
            continue
        try:
            # img = sitk.ReadImage(img_list[idx])
            nifti_img1 = nib.load(img_list[idx])

        except Exception as e:
            # skip corrupted images
            print(e)
            print("Image loading error:", imgname)
            continue

        img = nifti_img1.get_fdata()
        # img = sitk.GetArrayFromImage(img)
        try:
            img = resize_img(img)
        except Exception as e:  # Some images are corrupted
            print(e)
            print("Image resize error:", imgname)
            continue

        # preprocessed images are saved in numpy arrays
        np.save(OUTPUT_DATA_DIR + imgname.split('\\')[-1] + ".npy", img)

        # delete nii file after saving npy file
        os.remove(INPUT_DATA_DIR + imgname.split('\\')[-1])


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DATA_DIR):
        os.makedirs(OUTPUT_DATA_DIR)
    main()

#
# # Create an empty array to store intensity values from all images
# intensities = np.array([])
#
# # Loop through each image and extract intensity values
# for img_path in tqdm(img_list):
#     nifti_img = nib.load(img_path)
#     img = nifti_img.get_fdata()
#     intensities = np.append(intensities, img.flatten())
#
# plt.hist(intensities, bins=5)
# plt.show()
# exit()
