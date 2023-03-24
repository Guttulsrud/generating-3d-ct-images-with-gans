import itertools

import yaml
from model.train import train_network
from preprocessing.prepare_images import preprocess_images
from utils.Logger import Logger
from utils.storage.Storage import Storage

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if config['preprocessing']['enabled']:
        preprocess_images(config=config)

    with open('hpo.yaml', 'r') as f:
        hpo_config = yaml.safe_load(f)

    # Generate all combinations of hyperparameter values
    param_grid = list(
        itertools.product(hpo_config["generator_learning_rate"], hpo_config["generator_passes_per_iteration"],
                          hpo_config["discriminator_learning_rate"], hpo_config["encoder_learning_rate"]))

    # Print the total number of configurations
    print(f"Total configurations: {len(param_grid)}")

    for i, config_values in enumerate(param_grid):
        for j, key in enumerate(hpo_config.keys()):
            config['network'][key] = config_values[j]

        config['experiment_name'] = f"HPO_run_{i + 1}"

        storage = Storage(experiment_name=config['experiment_name'])
        logger = Logger(config=config)
        train_network(config=config, logger=logger)

        if config['upload_results']:
            storage.upload_results()
