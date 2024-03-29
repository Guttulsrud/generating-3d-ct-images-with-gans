import glob
import numpy as np
import yaml
from skimage.transform import resize
import os
import multiprocessing as mp
import nibabel as nib
from visualization.display_image import display_image

SUFFIX = '.nii.gz'
TRIM_BLANK_SLICES = True

# with open('train_data.json', 'r') as f:
#     train_data = yaml.safe_load(f)

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

    if data1.shape[1] != data2.shape[1] and data2.shape[1] == 401:
        data2 = np.delete(data2, 400, axis=1)

    if data1.shape[1] != data2.shape[1] and data2.shape[1] == 234:
        data2 = np.delete(data2, 232, axis=1)

    if data1.shape[0] != data2.shape[0] and data2.shape[0] == 234:
        data2 = np.delete(data2, 232, axis=0)

    if data1.shape[1] != data2.shape[1] and data2.shape[1] == 233:
        data2 = np.delete(data2, 232, axis=1)

    if data1.shape[0] != data2.shape[0] and data2.shape[0] == 233:
        data2 = np.delete(data2, 232, axis=0)
    try:
        concat = np.concatenate((data1, data2), axis=2)
        return concat
    except:
        print('Error')
        print(data1.shape)
        print(data2.shape)


def preprocess_images(config):
    print('Preprocessing images...')
    img_input_dir = config['preprocessing']['img_input_dir']
    mask_input_dir = config['preprocessing']['mask_input_dir']
    output_dir = config['preprocessing']['output_dir']
    num_jobs = config['preprocessing']['num_jobs']
    low_threshold = config['preprocessing']['low_threshold']
    high_threshold = config['preprocessing']['high_threshold']
    img_size = config['preprocessing']['img_size']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_list = list(glob.glob(img_input_dir + "*" + SUFFIX))
    mask_list = list(glob.glob(mask_input_dir + "*" + SUFFIX))

    img_list = zip(img_list, mask_list)
    processes = []

    for i in range(num_jobs):
        processes.append(
            mp.Process(target=preprocess,
                       args=(i, img_list, num_jobs, output_dir, low_threshold, high_threshold, img_size)))
    for p in processes:
        p.start()

    # Wait for processes to finish
    for p in processes:
        p.join()

    # All processes have finished, move on to the next section of code
    print("All processes have completed.")


def preprocess(batch_idx, img_list, num_jobs, output_dir, low_threshold, high_threshold, img_size):
    for idx, (image, mask) in enumerate(img_list):
        if idx % num_jobs != batch_idx:
            continue
        img_name = image.split('/')[-1]

        if not img_name.split('\\')[-1].replace('.gz', '') in train_data:
            continue

        if os.path.exists(output_dir + img_name.split('.')[0] + ".npy"):
            # skip images that already finished pre-processing
            continue

        nifti_img1 = nib.load(image)
        nifti_img2 = nib.load(mask)

        img = nifti_img1.get_fdata()
        mask = nifti_img2.get_fdata()

        img = np.interp(img, [low_threshold, high_threshold], [-1, 1])
        if TRIM_BLANK_SLICES:
            valid_plane_i = np.mean(img, (0, 1)) != -1  # Remove blank axial planes
            img = img[:, :, valid_plane_i]

        # mask = np.interp(mask, [0, 2], [-1, 1])
        mask = np.where((mask == 1) | (mask == 2), 1, -1)




        concat = alt_concat(img, mask)
        concat = resize(concat, (img_size, img_size, img_size), mode='constant', cval=1)
        # img = resize(img, (img_size, img_size, img_size // 2), mode='constant', cval=1)
        # mask = resize(mask, (img_size, img_size, img_size // 2), mode='constant', cval=1)

        #interpolated_resized_binary_2
        #img = concat[:, :, :img_size // 2]
        #mask = concat[:, :, img_size // 2:]


        # new_nifti_img = nib.Nifti1Image(concat, np.eye(4))
        # new_nifti_image = nib.Nifti1Image(img, np.eye(4))
        # new_nifti_mask = nib.Nifti1Image(mask, np.eye(4))

        # nib.save(new_nifti_img, f'../data/{img_size}/interpolated_resized/concat/' + img_name.split('.')[0].replace('images\\', '').replace('__CT', '') + ".nii.gz")
        # nib.save(new_nifti_image, f'../data/{img_size}/interpolated_resized/images/' + img_name.split('.')[0].replace('images\\', '').replace('__CT', '') + ".nii.gz")
        # nib.save(new_nifti_mask, f'../data/{img_size}/interpolated_resized/masks/' + img_name.split('.')[0].replace('images\\', '').replace('__CT', '') + ".nii.gz")
        np.save(output_dir + img_name.split('\\')[-1].replace('.nii.gz', '') + ".npy", concat)

        # os.remove(IMG_INPUT_DATA_DIR + img_name.split('\\')[-1])
        # os.remove(MASK_INPUT_DATA_DIR + mask_name.split('\\')[-1])

if __name__ == '__main__':
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    preprocess_images(config)
