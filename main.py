import time

import numpy as np

from config import config
import tensorflow as tf
import os

from utils.init import init
from utils.plotting import save_images

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

data_loader, logger, network = init()
import yaml


def apply_hparams():
    with open('hpo.yml', 'r') as f:
        return yaml.safe_load(f)


config = apply_hparams()

for epoch in range(1, config['training']['epochs'] + 1):
    print(f'Epoch: {epoch}', end='')
    logger.log(f'Epoch: {epoch}', newline=False)
    start = time.time()

    training_data = data_loader.get_dataset()

    for image_batch in training_data:
        print(image_batch.shape)
        network.train(images=image_batch, epoch=epoch)

    generated_image = network.generator(network.seed, training=False)
    generated_image = np.squeeze(generated_image)
    real_image = [np.squeeze(x) for x in training_data.take(1)][0]

    # save_images(real_image=real_image, fake_image=generated_image, epoch=epoch, path=network.path)

    # Save the model every 5 epochs
    if epoch % 5 == 0:
        network.save_checkpoint()

    print(f' - {round(time.time() - start, 2)} sec')
    logger.log(f' - {round(time.time() - start, 2)} sec')

# network.upload_tensorboard_results()
network.file_writer.flush()
