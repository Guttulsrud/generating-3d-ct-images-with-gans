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

    def get_dataset(self, batch_size, limit=None):
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.label_paths)).map(self.wrapper_load)

        if limit:
            dataset = dataset.take(limit)

        return dataset.batch(batch_size)


def resize_slice(slice, image_array):
    return tf.image.resize(slice, (image_array.shape[0] // 4, image_array.shape[1] // 4))