import yaml
from scipy.ndimage import gaussian_filter

from utils.DataLoader import DataLoader
from utils.config import load_config_file


class Augmentor:
    def __init__(self, save_images=True):
        with open('augmentations.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        self.data = DataLoader(data_type='train', config=self.config).get_dataset()

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
            print(image.shape)
            # image = tf.convert_to_tensor(image, dtype='float32')
            # image = tf.expand_dims(image, -1)
            # yield image


augmentor = Augmentor()
augmentor.apply_augmentation('gaussian_noise')
