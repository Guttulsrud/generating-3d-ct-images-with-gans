import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.inference.generate_image import generate_image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from visualization.display_image import display_image
import nibabel as nib
import matplotlib.pyplot as plt

run_search = True

if run_search:
    numpy_image = False
    image_paths = glob.glob('../data/cropped_original/images/*CT.nii.gz')
    mask_paths = glob.glob('../data/cropped_original/masks/*.nii.gz')
    data = []
    for image_path, mask_path in tqdm(zip(image_paths, mask_paths)):
        image = nib.load(image_path)
        image_data = image.get_fdata()
        # Get the header information
        header = image.header

        # Get the slope and intercept values
        slope = header.get('scl_slope', 1.0)
        intercept = header.get('scl_inter', 0.0)

        # Print the slope and intercept values
        print('Slope:', slope)
        print('Intercept:', intercept)
        exit()

        for x in image:
            for y in x:
                for z in y:
                    print(z)
                    exit()

        exit()
        image_min_pixel_value = min([min(x) for x in np.min(image_data, axis=0)])
        image_max_pixel_value = max([max(x) for x in np.max(image_data, axis=0)])

        mask = nib.load(image_path)
        mask = mask.get_fdata()
        mask_min_pixel_value = min([min(x) for x in np.min(mask, axis=0)])
        mask_max_pixel_value = max([max(x) for x in np.max(mask, axis=0)])

        if image_min_pixel_value != mask_min_pixel_value or image_max_pixel_value != mask_max_pixel_value:
            print('Error')
            print(image_min_pixel_value, image_max_pixel_value)
            print(mask_min_pixel_value, mask_max_pixel_value)
            exit()
        data.append({'low': image_min_pixel_value, 'high': image_max_pixel_value})
    exit()
    df = pd.DataFrame(data)
    df.to_csv('low_high.csv', index=False)

df = pd.read_csv('low_high.csv')

below_5000 = df[df['high'] <= 5000]
above_5000 = df[df['high'] > 5000]

# print(below_5000['high'].value_counts())
# print(df['high'].value_counts())

plt.hist(below_5000['high'], bins=10)
plt.title('Maximum value')
plt.xlabel('Value')
plt.ylabel('Frequency')
# plt.show()
plt.savefig('results/distribution_high.png')

# 28 of 524 images have a max value above 5000
# 340 of 524 images have max value of 3071, making it most common, next is 2976 with 114 images

# print(df['low'].value_counts())
plt.hist(df['low'], bins=10)
plt.title('Minimum value')
plt.xlabel('Value')
plt.ylabel('Frequency')
# plt.show()
plt.savefig('results/distribution_low.png')

# Most common is -2048 with 175, next is -1024 with 152, -3024 with 130

# Conclusion: Use 3071 as max value and -2048 as min value
