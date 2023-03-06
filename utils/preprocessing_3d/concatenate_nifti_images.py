import glob
import nibabel as nib
import os
from tqdm import tqdm

path = f'../../data'

image_paths = glob.glob(os.path.join(f'{path}/3d/preprocessed/images', '*CT.nii.gz'))
mask_paths = glob.glob(os.path.join(f'{path}/3d/preprocessed/masks', '*.nii.gz'))

for image_path, mask_path in tqdm(zip(image_paths, mask_paths)):
    path = os.path.join("../../data/3d/preprocessed/concatenated", image_path.split('images\\')[-1])
    path = path.replace('__CT', '')
    img1 = nib.load(image_path)
    img2 = nib.load(mask_path)

    if img1.shape != img2.shape:
        print(image_path, mask_path)
        continue

    concat_img = nib.concat_images([img1, img2])
    nib.save(concat_img, path)
