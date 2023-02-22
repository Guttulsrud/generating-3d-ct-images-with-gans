import numpy as np
from matplotlib import pyplot as plt
from monai.handlers.tensorboard_handlers import SummaryWriter
from config import config
from model.discriminator import build_discriminator
from model.generator import build_generator
import tensorflow as tf
import os
# from google.cloud import storage
from monai.visualize import plot_2d_or_3d_image


class WassersteinGPGAN:
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

        self.generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=generator_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=discriminator_learning_rate)

        self.checkpoint_prefix = os.path.join(f'{self.path}/training_checkpoints', "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        self.epoch = None

    @tf.function
    def gradient_penalty(self, images, generated_images):
        epsilon = tf.random.uniform([images.shape[0], 1, 1, 1], 0.0, 1.0)
        interpolated_images = epsilon * images + (1 - epsilon) * generated_images

        with tf.GradientTape() as t:
            t.watch(interpolated_images)
            interpolated_output = self.discriminator(interpolated_images, training=True)
        gradients = t.gradient(interpolated_output, interpolated_images)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
        return gradient_penalty

    @tf.function
    def train(self, images, epoch):
        self.epoch = epoch
        noise = tf.random.normal([1, *config['images']['shape']])

        # Train the discriminator
        with tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            disc_loss = discriminator_loss(real_output, fake_output)
            gradient_penalty = self.gradient_penalty(images, generated_images)
            disc_loss += config['network']['discriminator']['gradient_penalty_weight'] * gradient_penalty

        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = generator_loss(fake_output)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        self.log_scalars(gen_loss, disc_loss)
        self.log_images(generated_images)

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

    def log_images(self, images):
        with self.file_writer.as_default():
            img = tf.squeeze(images, axis=0)

            tf.summary.image("Generated Images", img, step=self.epoch)

    def log_scalars(self, gen_loss, disc_loss):
        with self.file_writer.as_default():
            tf.summary.scalar("Generator Loss", gen_loss, step=self.epoch)
            tf.summary.scalar("Discriminator Loss", disc_loss, step=self.epoch)


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


def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


def discriminator_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
