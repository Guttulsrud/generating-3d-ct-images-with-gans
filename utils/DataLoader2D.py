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

        self.image_paths = glob.glob(os.path.join(f'{path}/2d_padded/images', '*.png'))
        np.random.shuffle(self.image_paths)

    @staticmethod
    # define a function to load the PNG images
    def load_png(file_path):
        # load the image
        img = tf.io.read_file(file_path)
        # decode the PNG image
        img = tf.image.decode_png(img, channels=1)
        # convert the pixel values to floats in the range [0, 1]
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
        dataset = dataset.map(self.load_png)
        dataset = dataset.shuffle(buffer_size=10 * len(self.image_paths)).batch(self.config['dataloader']['batch_size'])

        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
