import os.path
from zipfile import ZipFile, error
from tqdm import tqdm


def unzip_file(file_path: str, destination: str):
    file_path = f'data/raw/{file_path}'
    file_path = find_file(file_path)
    if not file_path:
        raise Exception(f'Cannot find ZIP file for images')
    with ZipFile(file_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc=f'Unzipping {destination} files'):
            try:
                zip_ref.extract(member, f'../data/{destination}')
            except error as e:
                pass


def find_file(filename, current_dir='.'):
    """Find the specified file by going up the directory hierarchy."""
    # Construct the path to the file
    file_path = os.path.join(current_dir, filename)

    # Check if the file exists
    if os.path.exists(file_path):
        # Return the path to the file if it exists
        return os.path.normpath(file_path)

    # Go up one level in the directory hierarchy
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

    # Exit the function if we have reached the root directory
    if parent_dir == '/':
        return None

    # Recursively search for the file in the parent directory
    return find_file(filename, parent_dir)
