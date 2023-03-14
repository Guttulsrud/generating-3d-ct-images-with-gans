from model.train import train_network
from utils.config import get_config
from utils.logger import Logger

if __name__ == '__main__':
    config = get_config()
    logger = Logger(config=config)
    train_network(config=config, logger=logger)
