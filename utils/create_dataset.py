import glob
import os
import shutil

from tqdm import tqdm

original_images = sorted(glob.glob('../data/original_size/training/images/*CT.nii.gz'))
normalized_images = sorted(glob.glob('../data/augmented/normalized/images/*.nii.gz'))
reoriented_images = sorted(glob.glob('../data/augmented/reoriented/images/*.nii.gz'))
normalized_reoriented_images = sorted(glob.glob('../data/augmented/normalized_reoriented/images/*.nii.gz'))

output_folder = '../data/complete_dataset/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# copy original images
for image_path in tqdm(original_images):
    filename = os.path.basename(image_path)
    new_filename = f"{filename.split('.')[0].replace('__CT', '')}_original.nii.gz"
    output_path = os.path.join(output_folder, new_filename)
    shutil.copy(image_path, output_path)

# copy normalized images
for image_path in tqdm(normalized_images):
    filename = os.path.basename(image_path)
    new_filename = f"{filename.split('.')[0]}_normalized.nii.gz"
    output_path = os.path.join(output_folder, new_filename)
    shutil.copy(image_path, output_path)

# copy reoriented images
for image_path in tqdm(reoriented_images):
    filename = os.path.basename(image_path)
    new_filename = f"{filename.split('.')[0]}_reoriented.nii.gz"
    output_path = os.path.join(output_folder, new_filename)
    shutil.copy(image_path, output_path)

# copy normalized reoriented images
for image_path in tqdm(normalized_reoriented_images):
    filename = os.path.basename(image_path)
    new_filename = f"{filename.split('.')[0]}_normalized_reoriented.nii.gz"
    output_path = os.path.join(output_folder, new_filename)
    shutil.copy(image_path, output_path)