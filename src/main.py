import time
import tensorflow as tf
from src.data_loader import DataLoader
from src.model.discriminator import build_discriminator, discriminator_loss
from src.model.generator import build_generator, generator_loss
from src.utils.training import create_checkpoint, train_step
from src.utils.utils import generate_and_save_images

image_shape = (16, 16, 32)

generator = build_generator(input_shape=(*image_shape, 1))
discriminator = build_discriminator(input_shape=(*image_shape, 1))

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

seed = tf.random.normal([1, *image_shape])
EPOCHS = 5
training_data = DataLoader('training').get_dataset(batch_size=1, limit=3)
checkpoint, checkpoint_prefix = create_checkpoint(generator=generator,
                                                  discriminator=discriminator,
                                                  generator_optimizer=generator_optimizer,
                                                  discriminator_optimizer=discriminator_optimizer)

for epoch in range(1, EPOCHS + 1):
    print(f'Epoch: ', epoch)
    start = time.time()
    for image_batch in training_data:
        train_step(generator=generator,
                   discriminator=discriminator,
                   generator_loss=generator_loss,
                   discriminator_loss=discriminator_loss,
                   generator_optimizer=generator_optimizer,
                   discriminator_optimizer=discriminator_optimizer,
                   images=image_batch)

    generate_and_save_images(generator, epoch, seed)

    # Save the model every 5 epochs
    if epoch % 5 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print(f'Time for epoch {epoch + 1} is {time.time() - start} sec')
