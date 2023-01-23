import time
from src.data_loader import DataLoader
from src.model.network import Network

training_data = DataLoader('training').get_dataset(batch_size=1, limit=3)

network = Network()

EPOCHS = 5


for epoch in range(1, EPOCHS + 1):
    print(f'Epoch:', epoch)
    start = time.time()
    for image_batch in training_data:
        network.train(images=image_batch)

    network.save_images(epoch)

    # Save the model every 5 epochs
    if epoch % 5 == 0:
        network.save_checkpoint()

    print(f'Time for epoch {epoch + 1} is {time.time() - start} sec')