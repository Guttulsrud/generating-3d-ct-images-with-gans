import os.path
from zipfile import ZipFile, error
from tqdm import tqdm


def unzip_file(file_path: str, destination: str):
    if not os.path.exists(file_path):
        raise Exception(f'Cannot find ZIP file for images')

    with ZipFile(f'data/raw/{file_path}', 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc=f'Unzipping {destination} files'):
            try:
                zip_ref.extract(member, f'data/{destination}')
            except error as e:
                pass
