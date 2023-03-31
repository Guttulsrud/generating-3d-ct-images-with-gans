import glob
import os

from google.cloud import storage
from google.oauth2.service_account import Credentials


class Storage:
    def __init__(self, experiment_name):
        try:
            credentials = Credentials.from_service_account_file('utils/storage/thesis-377808-d22079a18bb5.json')
        except Exception:
            credentials = Credentials.from_service_account_file('storage/thesis-377808-d22079a18bb5.json')
        self.client = storage.Client(project='thesis-377808', credentials=credentials)
        self.bucket = self.client.bucket('thesis-tensorboard')
        self.experiment_name = experiment_name

    def upload_results(self):
        print(f'Uploading {self.experiment_name} to Cloud...')
        experiment_name = self.experiment_name
        blob = self.bucket.blob(f'{experiment_name}/config.yml')
        blob.upload_from_filename(f'logs/{experiment_name}/config.yml')

        event_file_path = glob.glob(f'logs/{experiment_name}/*events*')[0]

        blob = self.bucket.blob(event_file_path.replace('logs/', '').replace('\\', '/'))
        blob.upload_from_filename(event_file_path)

        for root, dirs, files in os.walk(f'logs/{experiment_name}/saved_model'):
            for index, filename in enumerate(files):
                print(f'Uploading checkpoint file ({index + 1}/{len(files)})')
                blob = self.bucket.blob(f'{experiment_name}/saved_model/{filename}')
                blob.upload_from_filename(f'logs/{experiment_name}/saved_model/{filename}')

    def download_results(self, experiment_name):
        print(f'Downloading {experiment_name} from Cloud...')

        path = '../saved_models/' + experiment_name
        if not os.path.exists(path):
            os.mkdir(path)

        blob = self.bucket.blob(f'{experiment_name}/config.yml')
        blob.download_to_filename(f'{path}/config.yml')

        for blob in self.bucket.list_blobs(prefix=f"{experiment_name}/"):
            if "event" in blob.name:
                f = blob.name.replace(experiment_name, path)
                blob.download_to_filename(f'../saved_models/{f}')

        # Download checkpoint files
        checkpoint_dir = f'{path}/saved_model'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        blobs = self.bucket.list_blobs(prefix=f'{experiment_name}/saved_model')
        for index, blob in enumerate(blobs):
            print(f'Downloading checkpoint file {index + 1}...')
            filename = os.path.join(checkpoint_dir, os.path.basename(blob.name))
            blob.download_to_filename(filename)

    def list_folders(self):
        folders = set()
        blobs = self.bucket.list_blobs()
        for blob in blobs:
            if '/' in blob.name:
                folder = '/'.join(blob.name.split('/')[:-1]) + '/'
                if folder.count('/') == 1:  # check if folder is a root folder
                    folders.add(folder.replace('/', ''))

        return sorted(list(folders))
