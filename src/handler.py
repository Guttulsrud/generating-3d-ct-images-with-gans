import logging
import os


class Handler:
    def __init__(self, start_datetime):
        os.mkdir(f'../logs/{start_datetime}')
        os.mkdir(f'../logs/{start_datetime}/epoch_images')
        self.logger = logging.getLogger('thesis')
        self.logger.setLevel(logging.DEBUG)
        self.logger.file_handler = logging.FileHandler(f'../logs/{start_datetime}/log.txt')
        self.logger.datefmt = '%Y-%m-%d %H:%M:%S'
        self.logger.formatter = logging.Formatter('%(asctime)s %(message)s')
        self.logger.filemode = 'a'

        # self.logger.basicConfig(filename=f'../logs/{start_datetime}/log.txt',
        #                     filemode='a',
        #                     format='%(asctime)s %(message)s',
        #                     datefmt='%H:%M:%S',
        #                     level=logging.DEBUG)

    def log(self, message):
        self.logger.info(message)
