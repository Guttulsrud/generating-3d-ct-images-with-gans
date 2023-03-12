import glob
import os
import shutil

from tqdm import tqdm

original_images = sorted(glob.glob('../data/original_size/images/*CT.nii.gz'))
normalized_images = sorted(glob.glob('../data/augmented/normalized/images/*.nii.gz'))
reoriented_images = sorted(glob.glob('../data/augmented/reoriented/images/*.nii.gz'))
normalized_reoriented_images = sorted(glob.glob('../data/augmented/normalized_reoriented/images/*.nii.gz'))

output_folder = '../data/complete_dataset/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def copy_files_from(folder, mode):
    for image_path in tqdm(folder):
        filename = os.path.basename(image_path)
        new_filename = f"{filename.split('.')[0]}_{mode}.nii.gz"
        output_path = os.path.join(output_folder, new_filename)
        shutil.copy(image_path, output_path)


# copy original images
for image_path in tqdm(original_images):
    filename = os.path.basename(image_path)
    new_filename = f"{filename.split('.')[0].replace('__CT', '')}_original.nii.gz"
    output_path = os.path.join(output_folder, new_filename)
    shutil.copy(image_path, output_path)

# copy normalized images
copy_files_from(normalized_images, 'normalized')

# copy reoriented images
copy_files_from(reoriented_images, 'reoriented')

# copy normalized reoriented images
copy_files_from(normalized_reoriented_images, 'normalized_reoriented')
