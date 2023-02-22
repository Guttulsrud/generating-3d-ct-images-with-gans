from model.WassersteinGAN import WassersteinGAN
from model.WassersteinGPGAN import WassersteinGPGAN
from utils.data_loader import DataLoader
from model.VanillaGAN import VanillaGAN
from config import config
from utils.logger import Logger
from datetime import datetime


def get_architecture():
    config_architecture = config['network']['architecture']

    architectures = {
        'vanilla': VanillaGAN,
        'wasserstein': WassersteinGAN,
        'wasserstein_gp': WassersteinGPGAN,
    }

    architecture = architectures.get(config_architecture)

    if not architecture:
        raise Exception('Invalid architecture. Valid are: vanilla, wasserstein')

    return architecture


def init():
    data_loader = DataLoader('training')

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H-%M-%S")

    logger = Logger(start_datetime=dt_string)
    architecture = get_architecture()

    network = architecture(start_datetime=dt_string)

    epochs = config['training']['epochs']

    print(f'Running {epochs} epochs with architecture: {config["network"]["architecture"]}')

    return data_loader, logger, network
