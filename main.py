import yaml

from model.train import train_network
from utils.Logger import Logger
from utils.storage.upload_results import upload_results
from google.oauth2.service_account import Credentials

credentials = Credentials.from_service_account_file('utils/storage/thesis-377808-d22079a18bb5.json')

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    logger = Logger(config=config)
    train_network(config=config, logger=logger)
    upload_results(experiment_name=f"logs/flip_rotate_gaus", credentials=credentials)
