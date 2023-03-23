import glob
import numpy as np
from skimage.transform import resize
import os
import multiprocessing as mp
import nibabel as nib
from visualization.display_image import display_image

NUM_JOBS = 8
IMG_SIZE = 128
IMG_INPUT_DATA_DIR = '../data/original/images/'
MASK_INPUT_DATA_DIR = '../data/original/masks/'
OUTPUT_DATA_DIR = '../data/npy/concatenated/'
LOW_THRESHOLD = -1024
HIGH_THRESHOLD = 600
SUFFIX = '.nii.gz'
TRIM_BLANK_SLICES = True


def resize_img(img, mask=False):
    nan_mask = np.isnan(img)
    img[nan_mask] = LOW_THRESHOLD
    img = np.interp(img, [LOW_THRESHOLD, HIGH_THRESHOLD], [-1, 1])

    if TRIM_BLANK_SLICES:
        valid_plane_i = np.mean(img, (0, 1)) != -1  # Remove blank axial planes
        img = img[:, :, valid_plane_i]

    img = resize(img, (IMG_SIZE, IMG_SIZE, IMG_SIZE / 2), mode='constant', cval=1 if mask else -1)

    return img


def concat_images(data1, data2):
    if data1.shape[1] != data2.shape[1] and data2.shape[1] == 513:
        data2 = np.delete(data2, 512, axis=1)

    if data1.shape[1] != data2.shape[1] and data2.shape[1] == 129:
        data2 = data2[:128, :128, :data2.shape[2]]

    if data1.shape[2] != data2.shape[2] and data2.shape[2] == 209:
        data2 = np.delete(data2, 208, axis=2)

    num_slices = data1.shape[2]
    concatenated_data = np.empty((data1.shape[0], data1.shape[1], num_slices * 2))

    for i in range(num_slices):
        concatenated_data[:, :, i * 2] = data1[:, :, i]
        concatenated_data[:, :, i * 2 + 1] = data2[:, :, i]

    return concatenated_data


def main():
    img_list = list(glob.glob(IMG_INPUT_DATA_DIR + "*" + SUFFIX))
    mask_list = list(glob.glob(MASK_INPUT_DATA_DIR + "*" + SUFFIX))

    img_list = zip(img_list, mask_list)
    processes = []
    for i in range(NUM_JOBS):
        processes.append(mp.Process(target=preprocess, args=(i, img_list)))
    for p in processes:
        p.start()


def preprocess(batch_idx, img_list):
    for idx, (image, mask) in enumerate(img_list):
        if idx % NUM_JOBS != batch_idx:
            continue
        img_name = image.split('/')[-1]
        mask_name = mask.split('/')[-1]

        if os.path.exists(OUTPUT_DATA_DIR + img_name.split('.')[0] + ".npy"):
            # skip images that already finished pre-processing
            continue
        try:
            nifti_img1 = nib.load(image)
            nifti_img2 = nib.load(mask)

        except Exception as e:
            # skip corrupted images
            print(e)
            print("Image loading error:", img_name, mask_name)
            continue

        img = nifti_img1.get_fdata()
        mask = nifti_img2.get_fdata()
        try:
            img = resize_img(img)
            mask = resize_img(mask, mask=True)
            concat = concat_images(img, mask)
        except Exception as e:
            print(e)
            print("Image resize error:", img_name, mask_name)
            continue

        np.save(OUTPUT_DATA_DIR + img_name.split('\\')[-1].replace('.nii.gz', '') + ".npy", concat)
        # os.remove(IMG_INPUT_DATA_DIR + img_name.split('\\')[-1])
        # os.remove(MASK_INPUT_DATA_DIR + mask_name.split('\\')[-1])


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DATA_DIR):
        os.makedirs(OUTPUT_DATA_DIR)
    main()