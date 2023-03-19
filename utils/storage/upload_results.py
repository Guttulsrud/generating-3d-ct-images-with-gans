import os

from google.cloud import storage
from google.oauth2.service_account import Credentials


def upload_results(experiment_name):
    credentials = Credentials.from_service_account_file('storage/thesis-377808-d22079a18bb5.json')
    client = storage.Client(project='thesis-377808', credentials=credentials)
    bucket = client.bucket('thesis-tensorboard')

    blob = bucket.blob(f'{experiment_name}/config.yml')
    blob.upload_from_filename(f'{experiment_name}/config.yml')

    for root, dirs, files in os.walk(f'{experiment_name}/checkpoint'):
        for index, filename in enumerate(files):
            print(f'Uploading checkpoint file ({index + 1}/{len(files)})')
            blob = bucket.blob(f'{experiment_name}/checkpoint/{filename}')
            blob.upload_from_filename(f'{experiment_name}/checkpoint/{filename}')


def download_results(experiment_name):
    print(f'Downloading {experiment_name} from Cloud...')
    credentials = Credentials.from_service_account_file('storage/thesis-377808-d22079a18bb5.json')
    client = storage.Client(project='thesis-377808', credentials=credentials)
    bucket = client.bucket('thesis-tensorboard')

    # Download config file
    path = '../saved_models/' + experiment_name
    if not os.path.exists(path):
        os.mkdir(path)
    blob = bucket.blob(f'{experiment_name}config.yml')
    blob.download_to_filename(f'{path}/config.yml')

    # Download checkpoint files
    checkpoint_dir = f'{path}/checkpoint'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    blobs = bucket.list_blobs(prefix=f'{experiment_name}checkpoint')
    for index, blob in enumerate(blobs):
        print(f'Downloading checkpoint file ({index + 1})')
        filename = os.path.join(checkpoint_dir, os.path.basename(blob.name))
        blob.download_to_filename(filename)


def list_folders():
    credentials = Credentials.from_service_account_file('storage/thesis-377808-d22079a18bb5.json')
    client = storage.Client(project='thesis-377808', credentials=credentials)
    bucket = client.get_bucket('thesis-tensorboard')

    folders = set()
    blobs = bucket.list_blobs()
    for blob in blobs:
        if '/' in blob.name:
            folder = '/'.join(blob.name.split('/')[:-1]) + '/'
            folders.add(folder)

    return sorted(list(folders))
