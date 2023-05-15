import SimpleITK as sitk
import numpy as np
from PIL import Image
import glob
import os

from scipy import ndimage
from tqdm import tqdm

images = glob.glob(os.path.join(f'../../data/2d_resampled/', '*.png'))
# masks = glob.glob(os.path.join(f'../../data/2d_resampled/masks', '*.png'))

for image_path in tqdm(images):
    image = Image.open(image_path)
    # mask = Image.open(mask_path)

    array1 = np.array(image)
    # array2 = np.array(mask)


    # m_part1, m_part2, m_part3, m_part4 = np.split(array2, 4, axis=0)

    # r1 = np.hstack((part1, m_part1))
    # r2 = np.hstack((part2, m_part2))
    # r3 = np.hstack((part3, m_part3))
    # r4 = np.hstack((part4, m_part4))
    final_image = np.hstack((np.split(array1, 8, axis=0)))

    result_image = Image.fromarray(final_image)
    result_image.save('../../data/stacked/' + image_path.split('\\')[1])