import numpy as np
import tensorflow as tf
import nibabel as nib
import glob
import os
from scipy.ndimage import gaussian_filter


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


class Augmentor:
    def __init__(self, config, save_images=True):
        self.config = config
        self.data = DataLoader(data_type='train', config=config).get_dataset()

    def apply_augmentation(self, augmentation):
        augmentations = {
            'gaussian_noise': self.gaussian_noise,
        }

        aug = augmentations.get(augmentation)
        if not aug:
            raise Exception(f'No augmentation found for {augmentation}')

        # aug(self.data)
        

    def gaussian_noise(self):
        sigma = self.config['augmentations']['gaussian_noise']['sigma']
        for image in self.data.get_dataset():
            # image = image.numpy()
            image = gaussian_filter(image, sigma=sigma)
            # image = tf.convert_to_tensor(image, dtype='float32')
            # image = tf.expand_dims(image, -1)
            # yield image
