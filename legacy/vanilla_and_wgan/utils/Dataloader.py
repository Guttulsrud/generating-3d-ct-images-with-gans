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

        image_path = config['dataloader']['image_path']
        self.image_paths = glob.glob(os.path.join(f'{path}/{image_path}/images', '*CT.nii.gz'))
        self.label_paths = glob.glob(os.path.join(f'{path}/{image_path}/labels', '*.nii.gz'))

        if not len(self.image_paths):
            raise Exception('No images found!')
        if not len(self.label_paths):
            raise Exception('No labels found!')

        if len(self.image_paths) != len(self.label_paths):
            raise Exception('Mismatched length images/labels')

        self.image_paths, self.label_paths = self._shuffle_filenames(self.image_paths, self.label_paths)

    @staticmethod
    def _shuffle_filenames(image_filenames, label_filenames):
        # Shuffle the order of the filenames
        shuffled_order = np.random.permutation(len(image_filenames))
        image_filenames = [image_filenames[i] for i in shuffled_order]
        label_filenames = [label_filenames[i] for i in shuffled_order]
        return image_filenames, label_filenames

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
        dataset = dataset.shuffle(len(self.image_paths))
        dataset = dataset.take(self.config['dataloader']['samples_per_epoch'])
        return dataset.batch(self.config['dataloader']['batch_size'])