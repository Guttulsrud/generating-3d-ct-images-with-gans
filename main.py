import itertools

import yaml
from model.train import train_network
# from preprocessing.prepare_images import preprocess_images
from preprocessing.prepare_images import preprocess_images
from utils.Logger import Logger
from utils.storage.Storage import Storage

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if config['preprocessing']['enabled']:
        preprocess_images(config=config)

    if not config['hpo']:
        storage = Storage(experiment_name=config['experiment_name'])
        logger = Logger(config=config)
        train_network(config=config, logger=logger)

        if config['upload_results']:
            storage.upload_results()

        exit()

    for i, size in enumerate([1280, 1536, 1792, 2048]):
        print(f'Running HPO configuration {i + 1}/4')
        config['experiment_name'] = f"H17_latent_dim{size}"
        config['latent_dim'] = size

        storage = Storage(experiment_name=config['experiment_name'])
        logger = Logger(config=config)
        train_network(config=config, logger=logger)

        if config['upload_results']:
            storage.upload_results()
