import SimpleITK as sitk
import numpy as np
from PIL import Image
import glob
import os

from scipy import ndimage
from tqdm import tqdm

images = glob.glob(os.path.join(f'../data/2d_squared', '*.png'))
max_x = 0
max_y = 0
max_shape = (23552, 2048)

def chop_image(image, slices):
    z_start = (image.shape[0] - slices) // 2
    z_end = z_start + slices

    return image[z_start:z_end:, :]

for image_path in tqdm(images):
    image = Image.open(image_path)

    array = np.array(image)
    x, y = array.shape

    # if x > max_x:
    #     max_x = x
    # if y > max_y:
    #     max_y = y
    # print('before ', array.shape)

    if x > 5000:

        array = chop_image(array, 5002)
    else:
        # Pad array
        num_rows_to_add = 5000 - x
        array = np.pad(array, ((num_rows_to_add, 0), (0, 0)), mode='constant')

    result_image = Image.fromarray(array)

    name = image_path.split('\\')[1]

    result_image.save(f'../data/2d_resampled_chopped/{name}')

# print(max_x, max_y)
