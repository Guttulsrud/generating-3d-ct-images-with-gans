import os
import yaml
from datetime import datetime


class Logger:
    def __init__(self, config):
        now = datetime.now()
        self.start_date_time = now.strftime("%m-%d %H-%M")

        if config['cluster_enabled']:
            path = f'/home/haakong/thesis/logs/{self.start_date_time}_{config["exp_name"]}'
        else:
            path = f'logs/{self.start_date_time}_{config["exp_name"]}'

        self.path = path
        os.mkdir(self.path)
        os.mkdir(f'{self.path}/checkpoint')
        with open(f'{self.path}/config.yml', 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

    def log(self, message, newline=True):
        with open(f'{self.path}/log.txt', "a") as log:
            if newline:
                log.write(f'{message}\n')
            else:
                log.write(f'{message}')
