import os

import torch
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
    RandAffined
)
import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os
import glob

with open('augmentations.yaml', 'r') as f:
    config = yaml.safe_load(f)


class Augmentor:
    def __init__(self):

        data_dir = config['input_dir']
        self.output_dir = config['output_dir']

        train_images = sorted(
            glob.glob(os.path.join(data_dir, "images", "*CT.nii.gz")))
        train_labels = sorted(
            glob.glob(os.path.join(data_dir, "masks", "*.nii.gz")))
        self.data = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels)
        ]
        self.loader = LoadImaged(keys=("image", "label"), image_only=False)

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

        image, mask = data_dict["image"], data_dict["label"]

        if len(image.shape) > 3:
            image = image[0, :, :]
            mask = mask[0, :, :]

        if display_image:
            self.create_plot(image, colormap='gray', title=title, mask=False)
        if display_mask:
            self.create_plot(mask, title=title)

    def add_channel(self, image):
        add_channel = EnsureChannelFirstd(keys=["image", "label"])
        return add_channel(image)

    def reorient_axes(self, image, display_image=False, display_mask=False):
        # Sometimes it is nice to have all the input volumes in a consistent axes orientation.
        # The default axis labels are Left (L), Right (R), Posterior (P), Anterior (A), Inferior (I), Superior (S).
        # The following transform is created to reorientate the volumes to have 'Posterior, Left,
        # Inferior' (PLI) orientation:
        orientation = Orientationd(keys=["image", "label"], axcodes="PLI")

        image = orientation(image)

        if display_image or display_mask:
            self.display_image(image, title='Reoriented Axes', display_image=display_image, display_mask=display_mask)
        return image

    def normalize(self, image, display_image=False, display_mask=False):
        # The input volumes might have different voxel sizes.
        # The following transform is created to normalise the volumes to have (1.5, 1.5, 5.) millimetre voxel size.

        # The transform is set to read the original voxel size information from `data_dict['image.affine']`,
        # which is from the corresponding NIfTI file, loaded earlier by `LoadImaged`.
        spacing = Spacingd(keys=["image", "label"], pixdim=(
            1.5, 1.5, 5.0), mode=("bilinear", "nearest"))

        image = spacing(image)

        if display_image or display_mask:
            self.display_image(image, title='Normalized Voxels', display_image=display_image, display_mask=display_mask)
        return image

    def random_affine_transformation(self, image, spatial_size, display_image=False, display_mask=False):
        # The following affine transformation is defined to output a (300, 300, 50) image patch.
        # The patch location is randomly chosen in a range of (-40, 40), (-40, 40), (-2, 2) in
        # x, y, and z axes respectively.

        # The translation is relative to the image centre.

        # The 3D rotation angle is randomly chosen from (-45, 45) degrees around
        # the z axis, and 5 degrees around x and y axes.

        # The random scaling factor is randomly chosen from (1.0 - 0.15, 1.0 + 0.15) along each axis.

        rand_affine = RandAffined(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=1.0,
            spatial_size=spatial_size,
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

    def save_image_mask(self, data_dict, folder_name):
        path = f'{self.output_dir}/{folder_name}'
        image_path = path + '/images'
        mask_path = path + '/masks'

        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.exists(image_path):
            os.makedirs(image_path)

        if not os.path.exists(mask_path):
            os.makedirs(mask_path)

        image_affine = data_dict['image'].affine
        mask_affine = data_dict['label'].affine

        image_name = data_dict['image'].meta['filename_or_obj'].split('\\')[-1]
        mask_name = data_dict['label'].meta['filename_or_obj'].split('\\')[-1]

        image_name = image_name.replace('__CT', '')

        writer = NibabelWriter()
        writer.set_data_array(data_dict['image'][0, :, :, :], channel_dim=None)
        writer.set_metadata({"affine": image_affine, "original_affine": image_affine})
        writer.write(f'{image_path}/{image_name}', verbose=False)

        writer = NibabelWriter()
        writer.set_data_array(data_dict['label'][0, :, :, :], channel_dim=None)
        writer.set_metadata({"affine": mask_affine, "original_affine": image_affine})
        writer.write(f'{mask_path}/{mask_name}', verbose=False)

    def load_image_mask(self, image_mask):
        image_mask = self.loader(image_mask)
        return self.add_channel(image_mask)


aug = Augmentor()
for image_mask_path in tqdm(aug.data):
    # image_mask = aug.load_image_mask(image_mask_path)
    image_mask = aug.loader(image_mask_path)

    image = image_mask['image']
    mask = image_mask['label']

    # Get the affine matrices of the two images
    affine1 = image_mask['image'].affine
    affine2 = image_mask['label'].affine

    # Check if the affine matrices are compatible
    if not np.allclose(affine1, affine2):
        print('The affine matrices of the two images are not compatible.')
        exit()

    # Get the number of slices in the images
    num_slices = image.shape[2]

    # Concatenate the two images slice by slice
    concatenated_data = np.empty((image.shape[0], image.shape[1], num_slices * 2))
    for i in range(num_slices):
        concatenated_data[:, :, i * 2] = image[:, :, i]
        concatenated_data[:, :, i * 2 + 1] = mask[:, :, i]

    # Update the affine matrix to reflect the concatenation
    # new_affine = np.copy(affine1)
    # new_affine[:3, 3] += affine1[:3, 2] * num_slices

    display_image(concatenated_data)
    exit()



