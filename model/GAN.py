import os
import tensorflow as tf

from model.discriminator_2d import build_discriminator
from model.generator_2d import build_generator


class GAN:
    def __init__(self, start_datetime, config):
        self.config = config
        self.seed = tf.random.normal([1, *config['images']['shape']])

        if config['cluster']['enabled']:
            self.path = f'/home/haakong/thesis/logs/{start_datetime}'
        else:
            self.path = f'logs/{start_datetime}'

        self.generator_learning_rate = config['network']['generator']['optimizer']['learning_rate']
        self.discriminator_learning_rate = config['network']['discriminator']['optimizer']['learning_rate']

        self.generator = build_generator(config)
        self.discriminator = build_discriminator(config)

        if config['inference']['enabled']:
            self.restore_checkpoint()
            return

        self.log_dir = os.path.join(f'{self.path}/tensorboard')
        self.file_writer = tf.summary.create_file_writer(self.log_dir)
        self.start_datetime = start_datetime
        self.checkpoint_prefix = None
        self.checkpoint = None
        self.epoch = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def init_checkpoint(self):
        self.checkpoint_prefix = os.path.join(f'{self.path}/training_checkpoints', "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    def save_checkpoint(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def restore_checkpoint(self):
        if self.config['inference']['enabled']:
            checkpoint_dir = self.config['inference']['checkpoint_path']
            checkpoint = tf.train.Checkpoint(generator=self.generator)
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        else:
            self.checkpoint.restore(tf.train.latest_checkpoint(f'{self.path}/training_checkpoints'))

    def log_images(self, images):
        with self.file_writer.as_default():
            print(images.shape)
            img = tf.squeeze(images, axis=0)

            tf.summary.image("Generated Images", img, step=self.epoch)

    def log_scalars(self, gen_loss, disc_loss):
        with self.file_writer.as_default():
            tf.summary.scalar("Generator Loss", gen_loss, step=self.epoch)
            tf.summary.scalar("Discriminator Loss", disc_loss, step=self.epoch)

    def generate_image(self):
        image = self.generator(self.seed, training=False)
        return image
