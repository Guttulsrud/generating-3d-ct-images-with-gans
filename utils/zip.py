from zipfile import ZipFile, error
from tqdm import tqdm


def unzip_file(file_path: str, destination: str):
    with ZipFile(f'data/raw/{file_path}', 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc='Extracting'):
            try:
                zip_ref.extract(member, f'data/{destination}')
            except error as e:
                pass
