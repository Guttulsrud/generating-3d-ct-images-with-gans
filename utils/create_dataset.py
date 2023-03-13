import glob
import os
import shutil

from tqdm import tqdm

normalized_images = sorted(glob.glob('../data/augmented/normalized_1.5x1.5x1.5/images/*.nii.gz'))
norm_translated = sorted(glob.glob('../data/augmented/norm_translated/images/*.nii.gz'))
norm_rotated = sorted(glob.glob('../data/augmented/norm_rotated/images/*.nii.gz'))
norm_reoriented = sorted(glob.glob('../data/augmented/norm_reoriented/images/*.nii.gz'))
norm_elastic_deformed = sorted(glob.glob('../data/augmented/norm_elastic_deformed/images/*.nii.gz'))
norm_affine_transformation = sorted(glob.glob('../data/augmented/norm_affine_transformation/images/*.nii.gz'))

output_folder = '../data/complete_dataset/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def copy_files_from(folder, mode):
    for image_path in tqdm(folder):
        filename = os.path.basename(image_path)
        new_filename = f"{filename.split('.')[0]}_{mode}.nii.gz"
        output_path = os.path.join(output_folder, new_filename)
        shutil.move(image_path, output_path)


copy_files_from(normalized_images, 'normalized')
copy_files_from(norm_translated, 'norm_translated')
copy_files_from(norm_rotated, 'norm_rotated')
copy_files_from(norm_reoriented, 'norm_reoriented')
copy_files_from(norm_elastic_deformed, 'norm_elastic_deformed')
copy_files_from(norm_affine_transformation, 'norm_affine_transformation')
