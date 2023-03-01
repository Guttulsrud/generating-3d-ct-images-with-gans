import SimpleITK as sitk
import numpy as np
from PIL import Image
import glob
import os

from scipy import ndimage
from tqdm import tqdm

images = glob.glob(os.path.join(f'../data/2d_resampled/concat', '*.png'))
max_x = 0
max_y = 0
max_shape = (23552, 2048)

for image_path in tqdm(images):
    image = Image.open(image_path)

    array = np.array(image)
    x, y = array.shape

    # if x > max_x:
    #     max_x = x
    # if y > max_y:
    #     max_y = y

    pad_width = ((0, max_shape[0] - array.shape[0]), (0, max_shape[1] - array.shape[1]))

    # pad the array
    padded_arr = np.pad(array, pad_width, mode='constant')
    result_image = Image.fromarray(padded_arr)

    name = image_path.split('\\')[1]

    result_image.save(f'../data/2d_padded/images/{name}')

print(max_x, max_y)
