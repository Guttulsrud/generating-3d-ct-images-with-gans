import time
from config import config
import tensorflow as tf
import os

from utils.init import init

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

data_loader, logger, network = init()


for epoch in range(1, config['training']['epochs'] + 1):
    print(f'Epoch: {epoch}', end='')
    logger.log(f'Epoch: {epoch}', newline=False)
    start = time.time()

    training_data = data_loader.get_dataset()

    for image_batch in training_data:
        network.train(images=image_batch, epoch=tf.convert_to_tensor(epoch, dtype=tf.int64))

    network.save_images(real_images=training_data)

    # Save the model every 5 epochs
    if epoch % 5 == 0:
        network.save_checkpoint()

    print(f' - {round(time.time() - start, 2)} sec')
    logger.log(f' - {round(time.time() - start, 2)} sec')

# network.upload_tensorboard_results()
network.file_writer.flush()
