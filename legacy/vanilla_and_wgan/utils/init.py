
from datetime import datetime

from legacy.dataloader.DataLoader2D import DataLoader2D
from legacy.model.VanillaGAN import VanillaGAN
from legacy.model.WassersteinGAN import WassersteinGAN
from legacy.model.WassersteinGPGAN import WassersteinGPGAN
from legacy.utils.logger import Logger


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
    data_loader = DataLoader2D(data_type='training', config=config)

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H-%M-%S")

    logger = Logger(start_datetime=dt_string, config=config)
    architecture = get_architecture(config=config)
    network = architecture(start_datetime=dt_string, config=config)

    epochs = config['training']['epochs']

    print(f'Running {epochs} epochs with architecture: {config["network"]["architecture"]}')

    return data_loader, logger, network