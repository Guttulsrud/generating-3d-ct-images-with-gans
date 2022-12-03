from zipfile import ZipFile

from utils.zip import unzip_file
from os import listdir
from os.path import isfile, join
import gzip
import shutil
import gzip
import os
from tqdm import tqdm


# unzip_file('training.zip', 'training')

def extract_nii_images_from_unzipped_files(image_type: str):
    if image_type not in ['training', 'testing']:
        raise Exception('Type must be either training or testing')

    images_path = f'data/unzipped/{image_type}/images'
    labels_path = f'data/unzipped/{image_type}/labels'

    if not os.path.exists(images_path):
        raise Exception(f'Cannot find ZIP file for {image_type} images')

    if not os.path.exists(labels_path):
        raise Exception(f'Cannot find ZIP file for {image_type} labels')

    folders = {
        'images': [f for f in listdir(images_path) if isfile(join(images_path, f)) if 'CT' in f],
        'labels': [f for f in listdir(labels_path) if isfile(join(labels_path, f)) if 'CT' in f]
    }

    for key, folder in folders.items():
        if not os.path.exists(f'data/processed/{image_type}/{key}'):
            os.makedirs(f'data/processed/{image_type}/{key}')

        for file in tqdm(folder, desc=f'Extracting {image_type} {key}'):
            file_path = images_path + '/' + file

            file = file.split('.gz')[0]
            with gzip.open(file_path, 'r') as f_in, open(f'data/processed/{image_type}/{key}/{file}', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


extract_nii_images_from_unzipped_files('training')
