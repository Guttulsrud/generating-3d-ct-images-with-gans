from os import listdir
from os.path import isfile, join
import shutil
import gzip
import os
from tqdm import tqdm

from utils.zip import unzip_file

image_type = 'training'

if image_type not in ['training', 'testing']:
    raise Exception('Type must be either training or testing')

images_path = f'../data/unzipped/{image_type}/images'
labels_path = f'../data/unzipped/{image_type}/labels'

if not os.path.exists(images_path) or not os.path.exists(labels_path):
    unzip_file(f'{image_type}.zip', image_type)

folders = {
    'images': [f for f in listdir(images_path) if isfile(join(images_path, f)) if 'CT' in f],
    'labels': [f for f in listdir(labels_path) if isfile(join(labels_path, f))]
}

for key, folder in folders.items():
    if not os.path.exists(f'../data/processed/{image_type}/{key}'):
        os.makedirs(f'../data/processed/{image_type}/{key}')

    for file in tqdm(folder, desc=f'Extracting {image_type} {key}'):
        if key == 'images':
            file_path = images_path + '/' + file
        else:
            file_path = labels_path + '/' + file

        file = file.split('.gz')[0]
        path = f'../data/processed/{image_type}/{key}/{file}'
        with gzip.open(file_path, 'r') as f_in, open(path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
