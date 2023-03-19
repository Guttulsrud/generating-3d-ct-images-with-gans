from model.train import train_network
from utils.config import get_config
from utils.logger import Logger
from utils.storage.upload_results import upload_results

if __name__ == '__main__':
    config = get_config()
    logger = Logger(config=config)
    train_network(config=config, logger=logger)
    upload_results(experiment_name=config['experiment_name'])
