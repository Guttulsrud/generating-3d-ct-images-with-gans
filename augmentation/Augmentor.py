import os
import time

from tqdm import tqdm

from visualization.display_image import display_image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import yaml
from monai.data import NibabelWriter
from monai.transforms import (
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Spacingd,
    RandAffined, Rand3DElasticd, LoadImage, FlipD
)
import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os
import glob

with open('augmentations.yaml', 'r') as f:
    config = yaml.safe_load(f)


def init():
    start_time_total = time.time()

    print('Augmenting images with: ', end='')
    for key, value in config.items():
        if key in ['images_only', 'input_dir', 'output_dir'] or value is False:
            continue
        print(f'{key.capitalize()}, ', end='')
    print('')
    return start_time_total


class Augmentor:
    def __init__(self):

        data_dir = config['input_dir']
        self.output_dir = config['output_dir']
        self.images_only = config['images_only']
        self.keys = ['image'] if self.images_only else ['image', 'mask']
        self.mode = 'bilinear' if self.images_only else ['bilinear', 'nearest']

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        train_images = sorted(
            glob.glob(os.path.join(data_dir, "images", "*CT.nii.gz")))

        if len(train_images) == 0:
            train_images = sorted(
                glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))

        if self.images_only:
            self.data = [
                {"image": image_name}
                for image_name in train_images
            ]
            self.loader = LoadImaged(keys="image", image_only=True)
            return

        train_labels = sorted(
            glob.glob(os.path.join(data_dir, "masks", "*.nii.gz")))

        self.data = [
            {"image": image_name, "mask": label_name}
            for image_name, label_name in zip(train_images, train_labels)
        ]
        self.loader = LoadImaged(keys=("image", "mask"), image_only=False)

    @staticmethod
    def create_plot(image, title, mask=True, colormap=None):
        n_slices = image.shape[2]
        n_cols = int(math.ceil(math.sqrt(n_slices)))
        n_rows = int(math.ceil(n_slices / float(n_cols)))

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 8))

        for i, ax in enumerate(axes.flat):
            if i < n_slices:
                ax.imshow(image[:, :, i], cmap=colormap)
                ax.axis('off')
            else:
                ax.set_visible(False)
        if mask:
            title = 'Mask: ' + title
        else:
            title = 'Image: ' + title

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    def display_image(self, data_dict, display_mask, display_image, title=None, load=False):
        if load:
            data_dict = self.loader(data_dict)

        if not title:
            title = "Original Image"

        if self.images_only:
            image = data_dict["image"]
            if len(image.shape) > 3:
                image = image[0, :, :]
        else:
            image, mask = data_dict["image"], data_dict["mask"]
            if len(image.shape) > 3:
                image = image[0, :, :]
                mask = mask[0, :, :]

        if display_image:
            self.create_plot(image, colormap='gray', title=title, mask=False)
        if display_mask and not self.images_only:
            self.create_plot(mask, title=title)

    def add_channel(self, image):
        add_channel = EnsureChannelFirstd(keys=self.keys)
        return add_channel(image)

    def reorient_axes(self, image, display_image=False, display_mask=False):
        # Sometimes it is nice to have all the input volumes in a consistent axes orientation.
        # The default axis labels are Left (L), Right (R), Posterior (P), Anterior (A), Inferior (I), Superior (S).
        # The following transform is created to reorientate the volumes to have 'Posterior, Left,
        # Inferior' (PLI) orientation:
        orientation = Orientationd(keys=self.keys, axcodes="PLI")

        image = orientation(image)

        if display_image or display_mask:
            self.display_image(image, title='Reoriented Axes', display_image=display_image, display_mask=display_mask)
        return image

    def flip(self, image, spatial_axis, display_image=False, display_mask=False):
        # For CT images of the neck and head, flipping along the x-axis
        # (left-right flipping) may not be the best choice because the
        # orientation of the brain and other structures in the head is
        # generally consistent and flipping the image could result in
        # anatomically incorrect images.

        # Instead, flipping along the y-axis (anterior-posterior flipping)
        # or z-axis (head-foot flipping) may be more appropriate, depending
        # on the orientation of the neck and head in the CT images. If the
        # images are acquired with the head in a supine position, flipping
        # along the z-axis (head-foot flipping) may be a good choice to
        # simulate the prone position, which could result in different
        # views of the neck and head structures.
        spacing = FlipD(keys=self.keys, spatial_axis=spatial_axis)

        image = spacing(image)

        if display_image or display_mask:
            self.display_image(image, title='Flipped', display_image=display_image, display_mask=display_mask)
        return image

    def normalize(self, image, voxels, display_image=False, display_mask=False):
        # The input volumes might have different voxel sizes.
        # The following transform is created to normalise the volumes to have (1.5, 1.5, 5.) millimetre voxel size.

        # The transform is set to read the original voxel size information from `data_dict['image.affine']`,
        # which is from the corresponding NIfTI file, loaded earlier by `LoadImaged`.

        spacing = Spacingd(keys=self.keys, pixdim=voxels, mode=self.mode)

        image = spacing(image)

        if display_image or display_mask:
            self.display_image(image, title='Normalized Voxels', display_image=display_image, display_mask=display_mask)
        return image

    def rotate(self, image, display_image=False, display_mask=False):
        # Define the rotation range
        min_angle = -20  # degrees
        max_angle = 20  # degrees
        rotate_range = (min_angle * np.pi / 180, max_angle * np.pi / 180, None)

        # Create the RandAffined transform with rotation only
        rand_affine = RandAffined(
            keys=self.keys,
            mode=self.mode,
            prob=1.0,
            spatial_size=image['image'].shape[1:],
            translate_range=None,
            rotate_range=rotate_range,
            scale_range=None,
            padding_mode="border",
        )
        image = rand_affine(image)

        if display_image or display_mask:
            self.display_image(image, title='Random rotation', display_mask=display_mask,
                               display_image=display_image)
        return image

    def random_translation(self, image, translate_range, display_image=False, display_mask=False):
        rand_affine = RandAffined(
            keys=self.keys,
            mode=self.mode,
            spatial_size=image['image'].shape[1:],
            translate_range=translate_range,
            padding_mode="border",
        )

        image = rand_affine(image)

        if display_image or display_mask:
            self.display_image(image, title='Random rotation', display_mask=display_mask,
                               display_image=display_image)
        return image

    def random_affine_transformation(self, image, display_image=False, display_mask=False):
        # The following affine transformation is defined to output a (300, 300, 50) image patch.
        # The patch location is randomly chosen in a range of (-40, 40), (-40, 40), (-2, 2) in
        # x, y, and z axes respectively.

        # The translation is relative to the image centre.

        # The 3D rotation angle is randomly chosen from (-45, 45) degrees around
        # the z axis, and 5 degrees around x and y axes.

        # The random scaling factor is randomly chosen from (1.0 - 0.15, 1.0 + 0.15) along each axis.

        rand_affine = RandAffined(
            keys=self.keys,
            mode=self.mode,
            prob=1.0,
            spatial_size=image['image'].shape[1:],
            translate_range=(40, 40, 2),
            rotate_range=(np.pi / 36, np.pi / 36, np.pi / 4),
            scale_range=(0.15, 0.15, 0.15),
            padding_mode="border",
        )

        image = rand_affine(image)

        if display_image or display_mask:
            self.display_image(image, title='Random Affine Transformation', display_mask=display_mask,
                               display_image=display_image)
        return image

    def random_elastic_deformation(self, image, display_image=False, display_mask=False):
        rand_elastic = Rand3DElasticd(
            keys=self.keys,
            mode=self.mode,
            prob=1.0,
            sigma_range=(5, 8),
            magnitude_range=(100, 200),
            spatial_size=image['image'].shape[1:],
            translate_range=(50, 50, 2),
            rotate_range=(np.pi / 36, np.pi / 36, np.pi),
            scale_range=(0.15, 0.15, 0.15),
            padding_mode="border",
        )
        rand_elastic.set_random_state(seed=123)
        image = rand_elastic(image)

        if display_image or display_mask:
            self.display_image(image, title='Random Elastic Deformation', display_mask=display_mask,
                               display_image=display_image)
        return image

    def save_image_mask(self, data_dict, folder_name):
        path = f'{self.output_dir}/{folder_name}'
        image_path = path + '/images'
        mask_path = path + '/masks'

        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.exists(image_path):
            os.makedirs(image_path)

        image_affine = data_dict['image'].affine
        image_name = data_dict['image'].meta['filename_or_obj'].split('\\')[-1]
        image_name = image_name.replace('__CT', '')

        writer = NibabelWriter()
        writer.set_data_array(data_dict['image'][0, :, :, :], channel_dim=None)
        writer.set_metadata({"affine": image_affine, "original_affine": image_affine})
        writer.write(f'{image_path}/{image_name}', verbose=False)

        if not self.images_only:
            if not os.path.exists(mask_path):
                os.makedirs(mask_path)

            mask_affine = data_dict['mask'].affine
            mask_name = data_dict['mask'].meta['filename_or_obj'].split('\\')[-1]
            writer = NibabelWriter()
            writer.set_data_array(data_dict['mask'][0, :, :, :], channel_dim=None)
            writer.set_metadata({"affine": mask_affine, "original_affine": image_affine})
            writer.write(f'{mask_path}/{mask_name}', verbose=False)

    def load_image_mask(self, image_mask):
        image_mask = self.loader(image_mask)
        return self.add_channel(image_mask)


aug = Augmentor()
start_time_total = init()

for index, image_mask_path in enumerate(aug.data):
    print(f'[{index + 1}/{len(aug.data)}]', end='')
    start_time = time.time()
    data = aug.load_image_mask(image_mask_path)

    remainder = len(aug.data) - index
    aug.display_image(data, display_image=True, display_mask=False)

    if config['flip']:
        normalized = aug.flip(data, spatial_axis=1, display_image=True)
        aug.save_image_mask(normalized, 'flipped_1')
        exit()
    if config['normalize']:
        normalized = aug.normalize(data, voxels=(1.5, 1.5, 1.5))
        aug.save_image_mask(normalized, 'normalized_1.5x1.5x1.5')

    if config['rotate']:
        rotated = aug.rotate(data, display_image=True)
        aug.save_image_mask(rotated, 'norm_rotated')

    if config['translate']:
        translated = aug.random_translation(data, translate_range=(40, 40, 2), display_image=True)
        aug.save_image_mask(translated, 'norm_translated')

    if config['elastic_deform']:
        elastic_deformed = aug.random_elastic_deformation(data)
        aug.save_image_mask(elastic_deformed, 'norm_elastic_deformed')

    if config['reorient']:
        reoriented = aug.reorient_axes(data)
        aug.save_image_mask(reoriented, 'norm_reoriented')

    if config['random_affine']:
        affine_transformation = aug.random_affine_transformation(data)
        aug.save_image_mask(affine_transformation, 'norm_affine_transformation')

    end_time = time.time()
    runtime_seconds = end_time - start_time_total
    runtime_minutes, runtime_seconds = divmod(runtime_seconds, 60)
    runtime_hours, runtime_minutes = divmod(runtime_minutes, 60)
    print(
        f' [It: {int(end_time - start_time)}s]'
        f' [Total: {float(runtime_hours)}h,'
        f' {int(runtime_minutes)}m,'
        f' {int(runtime_seconds)}s]'
        f' [Remaining: {float(runtime_minutes * remainder / 60)}h]')
