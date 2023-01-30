import numpy as np
import tensorflow as tf
import nibabel as nib
import glob
import os
from config import config


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

        max_shape = config['images']['padded_shape']
        padded_image = pad_image(image, max_shape)
        padded_label = pad_image(label, max_shape)

        concat = np.concatenate([padded_image, padded_label], 2)
        concat = tf.convert_to_tensor(concat, dtype='float32')
        concat = tf.expand_dims(concat, -1)

        return concat

    def get_dataset(self, batch_size, limit=None):
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.label_paths)).map(self.wrapper_load)

        if limit:
            dataset = dataset.take(limit)

        return dataset.batch(batch_size)


def pad_image(image, shape):
    return np.pad(image, [(0, shape[i] - image.shape[i]) for i in range(3)], 'constant', constant_values=0)
