import os

from google.cloud import storage
from google.oauth2.service_account import Credentials


def upload_results(experiment_name):
    credentials = Credentials.from_service_account_file('utils/storage/thesis-377808-d22079a18bb5.json')
    client = storage.Client(project='thesis-377808', credentials=credentials)
    bucket = client.bucket('thesis-tensorboard')

    blob = bucket.blob(f'{experiment_name}/config.yml')
    blob.upload_from_filename(f'{experiment_name}/config.yml')

    for root, dirs, files in os.walk(f'{experiment_name}/checkpoint'):
        for index, filename in enumerate(files):
            print(f'Uploading checkpoint file ({index + 1}/{len(files)})')
            blob = bucket.blob(f'{experiment_name}/checkpoint/{filename}')
            blob.upload_from_filename(f'{experiment_name}/checkpoint/{filename}')
