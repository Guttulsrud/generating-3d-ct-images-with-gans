import numpy as np
from matplotlib import pyplot as plt
from monai.handlers.tensorboard_handlers import SummaryWriter
from config import config
from model.discriminator import build_discriminator, discriminator_loss
from model.generator import build_generator, generator_loss
import tensorflow as tf
import os
# from google.cloud import storage
from monai.visualize import plot_2d_or_3d_image


class Network:
    def __init__(self, start_datetime, load_checkpoint=False):
        if config['cluster']['enabled']:
            self.path = f'/home/haakong/thesis/logs/{start_datetime}'
        else:
            self.path = f'logs/{start_datetime}'

        self.log_dir = os.path.join(f'{self.path}/tensorboard')
        self.file_writer = tf.summary.create_file_writer(self.log_dir)

        image_shape = config['images']['shape']
        self.start_datetime = start_datetime
        self.seed = tf.random.normal([1, *image_shape])

        if load_checkpoint:
            self.restore_checkpoint()
            return

        generator_learning_rate = config['network']['generator']['optimizer']['learning_rate']
        discriminator_learning_rate = config['network']['discriminator']['optimizer']['learning_rate']

        self.generator = build_generator()
        self.discriminator = build_discriminator()

        self.generator_optimizer = tf.keras.optimizers.Adam(generator_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(discriminator_learning_rate)

        self.checkpoint_prefix = os.path.join(f'{self.path}/training_checkpoints', "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

        # self.client = storage.Client(project='thesis-377808')
        # self.bucket = self.client.bucket('thesis-tensorboard')

    # This annotation causes the function to be "compiled" with TF.
    @tf.function
    def train(self, images, epoch):
        noise = tf.random.normal([1, *config['images']['shape']])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate synthetic image from noise with generator
            generated_images = self.generator(noise, training=True)

            # Get the predictions from the discriminator on the real and fake images
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Calculate loss for generator and discriminator
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

            self.log_scalars(epoch, gen_loss, disc_loss)
            self.log_images(epoch)

        # Get the gradients for each model
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Combine gradients with training variables
        generator_gradients = zip(gradients_of_generator, self.generator.trainable_variables)
        discriminator_gradients = zip(gradients_of_discriminator, self.discriminator.trainable_variables)

        # Apply gradients to the models
        self.generator_optimizer.apply_gradients(generator_gradients)
        self.discriminator_optimizer.apply_gradients(discriminator_gradients)

    def save_checkpoint(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def restore_checkpoint(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(f'{self.path}/training_checkpoints'))

    def save_images(self, epoch, real_images):
        generated_image = self.generator(self.seed, training=False)
        generated_image = np.squeeze(generated_image)
        real_image = [np.squeeze(x) for x in real_images.take(1)][0]

        os.mkdir(f'{self.path}/epochs/{epoch}')
        generated_plot = create_plot(generated_image, title=f'Epoch {epoch} - Generated image')
        real_plot = create_plot(real_image, title=f'Epoch {epoch} - Real image')

        generated_plot.savefig(f'{self.path}/epochs/{epoch}/generated.png')
        real_plot.savefig(f'{self.path}/epochs/{epoch}/real.png')

    def log_images(self, epoch):
        with self.file_writer.as_default():
            img = self.generator(self.seed, training=False)
            img = tf.squeeze(img, axis=0)

            tf.summary.image("Generated Images", img, step=epoch)

    def log_scalars(self, epoch, gen_loss, disc_loss):
        with self.file_writer.as_default():
            tf.summary.scalar("Generator Loss", gen_loss, step=epoch)
            tf.summary.scalar("Discriminator Loss", disc_loss, step=epoch)


# def upload_tensorboard_results(self):
#
#     for file in os.listdir(self.log_dir):
#         blob = self.bucket.blob(f'{self.start_datetime}/{file}')
#
#         blob.upload_from_filename(os.path.join(self.log_dir, file))
def separate_mask(input_image):
    image = input_image[:, :, :27]
    mask = input_image[:, :, 27:54]
    return image, mask


def create_plot(image, title):
    fig, axs = plt.subplots(9, 6, figsize=(15, 20))
    image, mask = separate_mask(image)

    for i in range(27):
        row = i // 3
        col = (i % 3) * 2

        axs[row, col].imshow(image[:, :, i], cmap='viridis')
        axs[row, col].set_title(f"Slice {i + 1}", size=15)
        axs[row, col].axis('off')

        axs[row, col + 1].imshow(mask[:, :, i], cmap='viridis')
        axs[row, col + 1].set_title(f"Slice {i + 1}", size=15)
        axs[row, col + 1].axis('off')

    fig.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig
