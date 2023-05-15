import os
import yaml


class Logger:
    def __init__(self, start_datetime, config):
        if config['cluster']['enabled']:
            path = f'/home/haakong/thesis/logs/{start_datetime}'
        else:
            path = f'logs/{start_datetime}'

        self.path = path
        os.mkdir(path)
        os.mkdir(f'{path}/epochs')
        with open(f'{path}/config.yml', 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

    def log(self, message, newline=True):
        with open(f'{self.path}/log.txt', "a") as log:
            if newline:
                log.write(f'{message}\n')
            else:
                log.write(f'{message}')