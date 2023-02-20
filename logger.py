import logging
import os
from config import config


class Logger:
    def __init__(self, start_datetime):
        if config['cluster']['enabled']:
            path = f'/home/haakong/thesis/logs/{start_datetime}'
        else:
            path = f'logs/{start_datetime}'

        self.path = path
        os.mkdir(path)
        os.mkdir(f'{path}/epoch_images')

    def log(self, message):
        with open(f'{self.path}/log.txt', "a") as log:
            log.write(f'{message}\n')
