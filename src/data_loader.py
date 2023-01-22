import numpy as np
import tensorflow as tf
import nibabel as nib
import glob
import os
import scipy as sp
from tqdm import tqdm


# Todo: use TF Records instead?

class DataLoader:
    def __init__(self, data_type):
        self.image_paths = glob.glob(os.path.join(f'../data/downsampled/10/{data_type}/images', '*CT.nii.gz'))
        self.label_paths = glob.glob(os.path.join(f'../data/downsampled/10/{data_type}/labels', '*.nii.gz'))

    def wrapper_load(self, img_path, label_path):
        return tf.py_function(func=self.preprocess_image_label, inp=[img_path, label_path],
                              Tout=tf.float32)

    @staticmethod
    def preprocess_image_label(image_path, label_path):
        image = nib.load(image_path.numpy().decode()).get_fdata()
        label = nib.load(label_path.numpy().decode()).get_fdata()

        # down_sampled_image = sp.ndimage.zoom(image, (0.25, 0.25, 0.25))
        # down_sampled_label = sp.ndimage.zoom(label, (0.25, 0.25, 0.25))

        max_shape = (16, 16, 16)
        padded_image = np.pad(image, [(0, max_shape[i] - image.shape[i]) for i in range(3)], 'constant',
                              constant_values=0)

        padded_label = np.pad(label, [(0, max_shape[i] - label.shape[i]) for i in range(3)], 'constant',
                              constant_values=0)

        concat = np.concatenate([padded_image, padded_label], 2)
        concat = tf.convert_to_tensor(concat, dtype='float32')
        concat = tf.expand_dims(concat, -1)

        return concat

    def get_dataset(self, batch_size, limit):
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.label_paths)).map(self.wrapper_load)
        dataset = dataset.take(limit)
        return dataset.batch(batch_size)


def resize_slice(slice, image_array):
    return tf.image.resize(slice, (image_array.shape[0] // 4, image_array.shape[1] // 4))


# ds = DataLoader('training').get_dataset()
#
# for x in ds:
#     print(x.shape)
#

def downsample():
    max_x = 0
    max_y = 0
    max_z = 0
    for image_path, label_path in tqdm(zip(glob.glob(os.path.join(f'../data/unzipped/training/images', '*CT.nii.gz')),
                                           glob.glob(os.path.join(f'../data/unzipped/training/labels', '*.nii.gz')))):
        image = nib.load(image_path)
        label = nib.load(label_path)
        image_path = image_path.split('\\')[1]
        label_path = label_path.split('\\')[1]

        image_data = image.get_fdata()
        label_data = label.get_fdata()

        down_sampled_image = sp.ndimage.zoom(image_data, (0.02, 0.02, 0.02))
        down_sampled_label = sp.ndimage.zoom(label_data, (0.02, 0.02, 0.02))

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

        nib.save(x, os.path.join("../data/downsampled/10/training/images", image_path))
        nib.save(y, os.path.join("../data/downsampled/10/training/labels", label_path))
        print(down_sampled_image.shape, down_sampled_label.shape)

    print(max_x, max_y, max_z)
    # 26 26 37 --- 5%