import yaml
import itertools
import pprint


def get_config():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if not config['hpo']:
        return config, range(1)

    with open('hpo.yaml', 'r') as f:
        hparams = yaml.safe_load(f)

    # Create a list of all possible hyperparameter combinations
    combinations = list(itertools.product(*hparams.values()))

    # Print the number of combinations to be tested
    print("Number of combinations to test: ", len(combinations))

    hparams = [dict(zip(hparams.keys(), combination)) for combination in combinations]
    return config, hparams


def apply_hpo(config, operation):
    if config['hpo']:
        config['network']['architecture'] = operation['architecture']
        config['dataloader']['samples_per_epoch'] = operation['samples_per_epoch']
    return config
