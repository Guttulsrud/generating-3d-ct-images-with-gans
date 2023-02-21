import os
import subprocess
from google.cloud import storage

client = storage.Client(project='thesis-377808')
bucket = client.bucket('thesis-tensorboard')

path = 'tensorboard_results'

if not os.path.exists(path):
    os.mkdir(path)

blobs = bucket.list_blobs()

for blob in blobs:
    folder_name = blob.name.split('/')[0]

    if os.path.exists(f'{path}/{folder_name}'):
        continue

    os.mkdir(f'{path}/{folder_name}')
    blob.download_to_filename(os.path.join(f'{path}', blob.name))
    subprocess.call(["tensorboard", "--logdir", f'{path}/{folder_name}'])
    break
