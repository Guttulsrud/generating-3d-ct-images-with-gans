import yaml
import itertools
import pprint


def load_config_file():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def get_config():
    config = load_config_file()

    # if not config['hpo']:
    #     return config, range(1)
    #
    # with open('hpo.yaml', 'r') as f:
    #     hparams = yaml.safe_load(f)
    #
    # # Create a list of all possible hyperparameter combinations
    # combinations = list(itertools.product(*hparams.values()))
    #
    # # Print the number of combinations to be tested
    # print("Number of combinations to test: ", len(combinations))
    #
    # hparams = [dict(zip(hparams.keys(), combination)) for combination in combinations]
    return config


def apply_hpo(config, operation):
    if config['hpo']:
        config['network']['architecture'] = operation['architecture']
        config['dataloader']['samples_per_epoch'] = operation['samples_per_epoch']
        config['network']['generator']['optimizer']['learning_rate'] = operation['generator_learning_rate']
        config['network']['discriminator']['optimizer']['learning_rate'] = operation['discriminator_learning_rate']

        resample_factor = operation['resample_factor']

        if resample_factor == 0.3:
            config['images']['shape'] = (154, 154, 54)
            config['dataloader']['image_path'] = 'chopped/0.3'
        elif resample_factor == 0.15:
            config['images']['shape'] = (78, 78, 78)
            config['dataloader']['image_path'] = 'chopped/0.15'
        elif resample_factor == 0.075:
            config['images']['shape'] = (38, 38, 38)
            config['dataloader']['image_path'] = 'chopped/0.075'
        else:
            raise Exception('Invalid resample factor')

    return config
