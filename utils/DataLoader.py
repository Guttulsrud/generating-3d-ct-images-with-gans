import numpy as np
import tensorflow as tf
import nibabel as nib
import glob
import os

class DataLoader:
    def __init__(self, data_type, config):
        self.config = config
        if config['cluster']['enabled']:
            path = f'/home/haakong/thesis/data'
        else:
            path = f'data'

        self.image_paths = glob.glob(os.path.join(f'{path}/chopped/resampled_0_03/images', '*CT.nii.gz'))
        self.label_paths = glob.glob(os.path.join(f'{path}/chopped/resampled_0_03/labels', '*.nii.gz'))

        if not len(self.image_paths):
            raise Exception('No images found!')
        if not len(self.label_paths):
            raise Exception('No labels found!')

        if len(self.image_paths) != len(self.label_paths):
            raise Exception('Mismatched length images/labels')

    def wrapper_load(self, img_path, label_path):
        return tf.py_function(func=self.preprocess_image_label, inp=[img_path, label_path],
                              Tout=tf.float32)

    @staticmethod
    def preprocess_image_label(image_path, label_path):
        image = nib.load(image_path.numpy().decode()).get_fdata()
        label = nib.load(label_path.numpy().decode()).get_fdata()

        concat = np.concatenate([image, label], 2)

        concat = tf.convert_to_tensor(concat, dtype='float32')
        concat = tf.expand_dims(concat, -1)

        return concat

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.label_paths)).map(self.wrapper_load)
        dataset = dataset.shuffle(self.config['dataloader']['samples_per_epoch'])
        dataset = dataset.take(self.config['dataloader']['samples_per_epoch'])
        return dataset.batch(self.config['dataloader']['batch_size'])