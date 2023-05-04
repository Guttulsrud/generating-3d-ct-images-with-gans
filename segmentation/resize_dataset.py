import glob
import numpy as np
import yaml
from scipy.ndimage import zoom
from skimage.transform import resize
import os
import multiprocessing as mp
import nibabel as nib
from visualization.display_image import display_image

SUFFIX = '.nii.gz'


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
        mask_name = mask.split('/')[-1]

        nifti_img1 = nib.load(image)
        nifti_img2 = nib.load(mask)

        image_data = nifti_img1.get_fdata()
        label_data = nifti_img2.get_fdata()
        label_data = np.where((label_data == 1) | (label_data == 2), 1, 0)

        desired_shape = np.array([256, 256, 128])

        # Calculate the zoom factors for the image and label data
        zoom_factors_image = desired_shape / np.array(image_data.shape)
        zoom_factors_label = desired_shape / np.array(label_data.shape)

        # Resize the image and label data using the zoom function
        image_data = zoom(image_data, zoom_factors_image, order=1)  # Linear interpolation for image data
        label_data = zoom(label_data, zoom_factors_label, order=0)  # Nearest-neighbor interpolation for label data

        nifti_image = nib.Nifti1Image(image_data, affine=np.eye(4))
        nifti_label = nib.Nifti1Image(label_data, affine=np.eye(4))

        nib.save(nifti_image,
                 '../data/resized/images/' + img_name.split('\\')[-1].replace('__CT', ''))
        nib.save(nifti_label,
                 '../data/resized/masks/' + mask_name.split('\\')[-1].replace('__CT', ''))


if __name__ == '__main__':

    if not os.path.exists('../data/resized/'):
        os.makedirs('../data/resized/')

    if not os.path.exists('../data/resized/images/'):
        os.makedirs('../data/resized/images/')

    if not os.path.exists('../data/resized/masks/'):
        os.makedirs('../data/resized/masks/')

    config = {
        'preprocessing': {
            'img_input_dir': '../data/256/normalized15mm/images/',
            'mask_input_dir': '../data/256/normalized15mm/masks/',
            'output_dir': '../data/resized/',
            'num_jobs': 4,
            'img_size': 256,
            'low_threshold': -1000,
            'high_threshold': 1000
        }
    }
    preprocess_images(config)
