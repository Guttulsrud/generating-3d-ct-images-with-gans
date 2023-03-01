import SimpleITK as sitk
import numpy as np
from PIL import Image
import glob
import os

from scipy import ndimage
from tqdm import tqdm

images = glob.glob(os.path.join(f'../data/2d_resampled/images', '*.png'))
masks = glob.glob(os.path.join(f'../data/2d_resampled/masks', '*.png'))
images = images[:100]
masks = masks[:100]
for image_path, mask_path in tqdm(zip(images, masks)):
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    array1 = np.array(image)
    array2 = np.array(mask)

    part1, part2, part3, part4 = np.split(array1, 4, axis=0)
    m_part1, m_part2, m_part3, m_part4 = np.split(array2, 4, axis=0)

    r1 = np.hstack((part1, m_part1))
    r2 = np.hstack((part2, m_part2))
    r3 = np.hstack((part3, m_part3))
    r4 = np.hstack((part4, m_part4))
    final_image = np.hstack((r1, r2, r3, r4))

    result_image = Image.fromarray(final_image)
    result_image.save('../data/2d_resampled/concat/' + image_path.split('\\')[1])