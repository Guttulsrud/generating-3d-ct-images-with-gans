from train import train
from utils.config import get_config
from utils.logger import Logger

if __name__ == '__main__':
    config = get_config()
    logger = Logger(config=config)
    train(logger=logger, config=config)
