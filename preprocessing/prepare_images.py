import glob
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize, rescale
import os
import multiprocessing as mp
import nibabel as nib
from visualization.display_image import display_image

NUM_JOBS = 8
IMG_SIZE = 256
IMG_INPUT_DATA_DIR = '../data/cropped_original/images/'
MASK_INPUT_DATA_DIR = '../data/cropped_original/masks/'
OUTPUT_DATA_DIR = '../data/npy/256/'
LOW_THRESHOLD = -1024
HIGH_THRESHOLD = 600
SUFFIX = '.nii.gz'
TRIM_BLANK_SLICES = True


def plot_center_sagittal_slice(data):
    data = np.transpose(data, axes=(1, 0, 2))
    center_sagittal_index = data.shape[1] // 2
    center_sagittal_slice = data[:, center_sagittal_index, :]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(np.rot90(center_sagittal_slice), cmap='gray', aspect='equal',
                   extent=[0, data.shape[0], 0, data.shape[0]])
    ax.set_title(f'Center Sagittal (slice {center_sagittal_index})')
    ax.axis('off')
    plt.show()


def resize_img(img):
    img = np.interp(img, [LOW_THRESHOLD, HIGH_THRESHOLD], [-1, 1])

    if TRIM_BLANK_SLICES:
        valid_plane_i = np.mean(img, (0, 1)) != -1  # Remove blank axial planes
        img = img[:, :, valid_plane_i]

    img = resize(img, (IMG_SIZE, IMG_SIZE, IMG_SIZE / 2), mode='constant', cval=1)

    return img


def resize_mask(img):
    img = np.interp(img, [0, 2], [-1, 1])
    img = resize(img, (IMG_SIZE, IMG_SIZE, IMG_SIZE / 2), mode='constant', cval=1)

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


def alt_concat(data1, data2):
    if data1.shape[1] != data2.shape[1] and data2.shape[1] == 513:
        data2 = np.delete(data2, 512, axis=1)

    if data1.shape[1] != data2.shape[1] and data2.shape[1] == 129:
        data2 = data2[:128, :128, :data2.shape[2]]

    if data1.shape[2] != data2.shape[2] and data2.shape[2] == 209:
        data2 = np.delete(data2, 208, axis=2)

    num_slices = data1.shape[2]
    concatenated_data = np.empty((data1.shape[0], data1.shape[1], num_slices * 2))

    return np.concatenate((data1, data2), axis=2)


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

        img = np.interp(img, [LOW_THRESHOLD, HIGH_THRESHOLD], [-1, 1])
        if TRIM_BLANK_SLICES:
            valid_plane_i = np.mean(img, (0, 1)) != -1  # Remove blank axial planes
            img = img[:, :, valid_plane_i]

        mask = np.interp(mask, [0, 2], [-1, 1])

        concat = alt_concat(img, mask)
        concat = resize(concat, (IMG_SIZE, IMG_SIZE, IMG_SIZE), mode='constant', cval=-1)

        np.save(OUTPUT_DATA_DIR + img_name.split('\\')[-1].replace('.nii.gz', '') + ".npy", concat)
        # os.remove(IMG_INPUT_DATA_DIR + img_name.split('\\')[-1])
        # os.remove(MASK_INPUT_DATA_DIR + mask_name.split('\\')[-1])


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DATA_DIR):
        os.makedirs(OUTPUT_DATA_DIR)
    main()
