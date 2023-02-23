import yaml


def get_config():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if not config['hpo']:
        return config

    with open('hpo.yaml', 'r') as f:
        hparams = yaml.safe_load(f)

    # do something
