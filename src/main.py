import time
from src.data_loader import DataLoader
from src.model.network import Network
from config import config
from handler import Handler
from datetime import datetime

data_loader = DataLoader('training')
training_data = data_loader.get_dataset(batch_size=config['dataloader']['batch_size'],
                                        limit=config['dataloader']['limit'])

now = datetime.now()
dt_string = now.strftime("%Y-%m-%d %H-%M-%S")

handler = Handler(start_datetime=dt_string)
network = Network(start_datetime=dt_string)

epochs = config['training']['epochs']


for epoch in range(1, epochs + 1):
    print(f'Running epoch:', epoch)
    handler.log(f'Running epoch: {epoch}')
    start = time.time()
    for image_batch in training_data:
        network.train(images=image_batch)

    network.save_images(epoch)

    # Save the model every 5 epochs
    if epoch % 5 == 0:
        network.save_checkpoint()

    print(f'Time for epoch {epoch} is {time.time() - start} sec')
    handler.log(f'Time for epoch {epoch} is {time.time() - start} sec')
