import glob
import os

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from visualization.display_image import display_image


#


# image_paths = glob.glob('../data/original/images/*CT.nii.gz')
# mask_paths = glob.glob('../data/original/masks/*.nii.gz')
# data = []
# for image_path, mask_path in tqdm(zip(image_paths, mask_paths)):
#     image = nib.load(image_path)
#
#     if 'CHUM' not in image_path:
#         continue
#
#     print(image_path,'   ' , image.shape)
#     # img_data = image.get_fdata()
def plot_center_sagittal_slice(name, data):
    data = np.transpose(data, axes=(1, 0, 2))
    center_sagittal_index = data.shape[1] // 2
    center_sagittal_slice = data[:, center_sagittal_index, :]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(np.rot90(center_sagittal_slice), cmap='gray', aspect='equal',
                   extent=[0, data.shape[0], 0, data.shape[0]])
    ax.set_title(f'{name}')
    ax.axis('off')
    plt.show()


single = False

if single:

    source = 'MDA'

    image_paths = glob.glob(f'../data/original/images/*{source}*CT.nii.gz')
    mask_paths = glob.glob(f'../data/original/masks/*{source}*.nii.gz')

else:
    image_paths = glob.glob('../data/original/images/*CT.nii.gz')
    mask_paths = glob.glob('../data/original/masks/*.nii.gz')
data = []

if not os.path.exists('../data/cropped'):
    os.mkdir('../data/cropped')

if not os.path.exists('../data/cropped/images'):
    os.mkdir('../data/cropped/images')

if not os.path.exists('../data/cropped/masks'):
    os.mkdir('../data/cropped/masks')

images_to_crop = {
    'CHUM': {
        'ids': [
            '013',
            '035',
            '040',
            '065',
        ],
        'crop': 172
    },
    'CHUP': {
        'ids': 'all',
        'crop': 340
    },
    'CHUS': {
        'ids': ['101'],
        'crop': 100
    },
    'CHUV': {
        'ids': ['002',
                '003',
                '004',
                '005',
                '006',
                '007',
                '008',
                '009',
                '010',
                '011',
                '022',
                '023',
                '024',
                '025',
                '026',
                '027',
                '028',
                '029',
                '031',
                '032',
                '034',
                '035',
                '038',
                '040',
                '041'
                '043',
                '044',
                '045',
                '046',
                '049',
                '050',
                '051',
                '052'],
        'crop': 172
    },
    'HMR': {
        'ids': [
            '001',
            '012',
            '013',
            '028',
        ],
        'crop': 172
    },
    'MDA': {
        'ids': [
            '001',
            '003',
            '004',
            '005',
            '006',
            '007',
            '029',
            '030',
            '032',
            '036',
            '048',
            '052',
            '055',
            '056',
            '057',
            '059',
            '063',
            '067',
            '072',
            '074',
            '075',
            '086',
            '087',
            '092',
            '097',
            '101',
            '102',
            '103',
            '108',
            '109',
            '111',
            '117',
            '120',
            '122',
            '123',
            '144',
            '153',
            '157',
            '161',
            '163',
            '164',
            '170',
            '175',
            '178',
            '188',
            '190',
            '194',
            '200'
        ],
        'crop': 140
    },
}

for image_path, mask_path in tqdm(zip(image_paths, mask_paths)):
    img = nib.load(image_path)
    mask = nib.load(mask_path)

    source = image_path.split('images\\')[1].split('-')[0]
    image_id = image_path.split(f'images\\{source}-')[1].replace('__CT.nii.gz', '')

    imgs = images_to_crop.get(source)

    if not imgs:
        continue

    if image_id not in imgs['ids'] and imgs['ids'] != 'all':
        continue

    # img_data = img.get_fdata()
    # y = img_data.shape[2]
    # print(img_data.shape)
    # plot_center_sagittal_slice('cropped', img_data)

    # plot_center_sagittal_slice('original', img_data)
    # img_data = img_data[:, :, imgs['crop']:y]
    # print(img_data.shape)
    # plot_center_sagittal_slice(image_path, img_data)
    # continue

    y = img.shape[2]
    cropped_img = img.slicer[:, :, imgs['crop']:y]
    cropped_mask = mask.slicer[:, :, imgs['crop']:y]

    cropped_img.to_filename(f'../data/cropped/images/{source}-{image_id}__CT.nii.gz')
    cropped_mask.to_filename(f'../data/cropped/masks/{source}-{image_id}.nii.gz')

    # img_data = img.get_fdata()
    # y = img_data.shape[2]
    # plot_center_sagittal_slice('original', img_data)
    # img_data = img_data[:, :, 170:y]
    # # img_data = img_data[:, :, y - 248:y]
    # plot_center_sagittal_slice('cropped', img_data)
