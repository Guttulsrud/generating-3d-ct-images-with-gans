import SimpleITK as sitk
import numpy as np
from PIL import Image
import glob
import os

from scipy import ndimage
from tqdm import tqdm
import cv2  # for reading and displaying images

images = glob.glob(os.path.join(f'../data/2d_resampled_chopped', '*.png'))

for image_path in tqdm(images):
    image = Image.open(image_path)
    image = np.array(image)
    height, width = image.shape[:2]
    num_splits = 6
    split_height = height // num_splits
    parts = [image[i * split_height:(i + 1) * split_height] for i in range(num_splits)]
    final_image = np.hstack(parts)
    result_image = Image.fromarray(final_image)
    result_image.save('../data/2d_squared/' + image_path.split('\\')[1])
