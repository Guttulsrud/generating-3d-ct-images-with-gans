import yaml
from model.train import train_network
from utils.Logger import Logger
from utils.storage.Storage import Storage

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    storage = Storage(experiment_name=config['experiment_name'])
    logger = Logger(config=config)
    train_network(config=config, logger=logger)

    if config['upload_results']:
        storage.upload_results()

