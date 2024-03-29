import numpy as np
import tensorflow as tf
import nibabel as nib
import glob
import os
from PIL import Image


class DataLoader2D:
    def __init__(self, data_type, config):
        self.config = config
        if config['cluster']['enabled']:
            path = f'/home/haakong/thesis/data'
        else:
            path = f'data'

        self.image_paths = glob.glob(f'{path}/stacked/*.png')
        print(f'Found {len(self.image_paths)} {data_type} images')
        np.random.shuffle(self.image_paths)

    @staticmethod
    # define a function to load the PNG images
    def load_png(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
        dataset = dataset.map(self.load_png)
        dataset = dataset.shuffle(buffer_size=len(self.image_paths)).batch(self.config['dataloader']['batch_size'])

        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
