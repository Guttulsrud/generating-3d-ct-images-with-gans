import time
from data_loader import DataLoader
from model.network import Network
from config import config
from logger import Logger
from datetime import datetime
import tensorflow as tf
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


data_loader = DataLoader('training')
training_data = data_loader.get_dataset(batch_size=config['dataloader']['batch_size'],
                                        limit=config['dataloader']['limit'])

now = datetime.now()
dt_string = now.strftime("%Y-%m-%d %H-%M-%S")

logger = Logger(start_datetime=dt_string)
network = Network(start_datetime=dt_string)

# network.upload_tensorboard_results()
#
# exit()
epochs = config['training']['epochs']

for epoch in range(1, epochs + 1):
    print(f'Epoch: {epoch}')
    logger.log(f'Epoch: {epoch}')
    start = time.time()

    for image_batch in training_data:
        network.train(images=image_batch, epoch=tf.convert_to_tensor(epoch, dtype=tf.int64))

    network.save_images(epoch)

    # Save the model every 5 epochs
    if epoch % 5 == 0:
        network.save_checkpoint()

    print(f'Time for epoch {epoch}: {time.time() - start} sec')
    logger.log(f'Time for epoch {epoch}: {time.time() - start} sec')

network.file_writer.flush()
