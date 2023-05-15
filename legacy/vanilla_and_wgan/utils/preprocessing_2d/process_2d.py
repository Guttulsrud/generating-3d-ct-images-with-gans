import numpy as np
from PIL import Image
import glob
import os
from scipy import ndimage
from tqdm import tqdm


def stack():
    images = glob.glob(f'../../data/2d_resampled/*.png')

    for image_path in tqdm(images):
        try:
            image = Image.open(image_path)
        except:
            print(image_path)
            continue
        image = np.array(image)

        x, y = image.shape

        # if x < 121344:
        #     image = np.pad(image, pad_width=((0, 121344 - x), (0, 0)), mode='constant')
        # else:
        min_first_dim = 121344
        chop_amount = max(0, image.shape[0] - min_first_dim)
        pad_amount = max(0, min_first_dim - image.shape[0])
        chopped_image = image[chop_amount:, :]
        # image = np.pad(chopped_image, pad_width=((0, pad_amount), (0, 0)), mode='constant')


        final_image = np.hstack(np.split(chopped_image, 10, axis=0))
        result_image = Image.fromarray(final_image)
        result_image.save('../../data/stacked/' + image_path.split('\\')[1])

        # array1 = np.array(image)
        # array2 = np.array(mask)
        #
        # part1, part2, part3, part4 = np.split(array1, 4, axis=0)
        # m_part1, m_part2, m_part3, m_part4 = np.split(array2, 4, axis=0)
        #
        # r1 = np.hstack((part1, m_part1))
        # r2 = np.hstack((part2, m_part2))
        # r3 = np.hstack((part3, m_part3))
        # r4 = np.hstack((part4, m_part4))
        # final_image = np.hstack((r1, r2, r3, r4))
        #
        # result_image = Image.fromarray(final_image)
        # result_image.save('../data/2d_resampled/concat/' + image_path.split('\\')[1])


def resample_2d_png(percent):
    images = glob.glob(os.path.join(f'../../data/2d/images', '*.png'))
    for image_path in tqdm(images):
        try:
            image = Image.open(image_path)
        except:
            print(image_path)
        image = np.array(image)

        resampled_image = ndimage.zoom(image, (percent, percent))
        result_image = Image.fromarray(resampled_image)
        result_image.save('../../data/2d_resampled/' + image_path.split('\\')[1])


# resample_2d_png(percent=0.5)
stack()
