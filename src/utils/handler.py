import logging
import os


class Handler:
    def __init__(self, start_datetime):

        os.mkdir(start_datetime)

        logging.basicConfig(filename=f'logs/{start_datetime}/log.txt',
                            filemode='a',
                            format='%(asctime)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

    @staticmethod
    def log(message):
        logging.info(message)
