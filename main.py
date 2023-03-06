import time
import numpy as np
from utils.config import get_config, apply_hpo
from utils.init import init
from utils.plotting import save_images
import os
import tensorflow as tf
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

config, operations = get_config()

for index, operation in enumerate(operations):

    if config['hpo']:
        print(f'Running HPO config: {index + 1}/{len(operations)}')

    config = apply_hpo(config, operation)
    data_loader, logger, network = init(config)
    training_data = data_loader.get_dataset()

    for epoch in range(1, config['training']['epochs'] + 1):

        print(f'Epoch: {epoch}', end='')
        logger.log(f'Epoch: {epoch}', newline=False)
        start = time.time()

        for image_batch in training_data:
            network.train(images=image_batch, epoch=epoch)

        generated_image = network.generator(network.seed, training=False)
        generated_image = np.squeeze(generated_image)
        real_image = [np.squeeze(x) for x in training_data.take(1)][0]

        save_images(real_image=real_image, fake_image=generated_image, epoch=epoch, path=network.path, config=config)

        # Save the model every 5 epochs
        if epoch % 5 == 0:
            network.save_checkpoint()

        print(f' - {round(time.time() - start, 2)} sec')
        logger.log(f' - {round(time.time() - start, 2)} sec')
