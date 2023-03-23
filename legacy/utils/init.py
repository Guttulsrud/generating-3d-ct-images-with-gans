from dataloader.DataLoader import DataLoader
from inference.WassersteinGAN import WassersteinGAN
from inference.WassersteinGPGAN import WassersteinGPGAN
from inference.VanillaGAN import VanillaGAN
from utils.Logger import Logger
from datetime import datetime


def get_architecture(config):
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


def init(config):
    data_loader = DataLoader(data_type='training', config=config)

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H-%M-%S")

    logger = Logger(start_datetime=dt_string, config=config)
    architecture = get_architecture(config=config)
    network = architecture(start_datetime=dt_string, config=config)

    epochs = config['training']['epochs']

    print(f'Running {epochs} epochs with architecture: {config["network"]["architecture"]}')

    return data_loader, logger, network
