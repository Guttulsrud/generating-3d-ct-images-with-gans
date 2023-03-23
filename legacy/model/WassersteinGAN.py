from model.GAN import GAN
from model.discriminator import build_discriminator
from model.generator import build_generator
import tensorflow as tf
import os
from model.loss_functions import wasserstein_discriminator_loss, wasserstein_generator_loss


class WassersteinGAN(GAN):
    def __init__(self, start_datetime, config):
        super().__init__(start_datetime, config)

        self.generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.generator_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.discriminator_learning_rate)

        self.clip_value = config['network']['discriminator']['clip_value']
        self.init_checkpoint()

    @tf.function
    def train(self, images, epoch):
        self.epoch = epoch
        noise = tf.random.normal([images.shape[0], *self.config['images']['shape']])

        with tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            disc_loss = wasserstein_discriminator_loss(real_output, fake_output)

        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        # Clip the weights of the discriminator after each training step
        for variable in self.discriminator.trainable_variables:
            variable.assign(tf.clip_by_value(variable, -self.clip_value, self.clip_value))

        # Train the generator
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = wasserstein_generator_loss(fake_output)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        self.log_scalars(gen_loss, disc_loss)
        self.log_images(generated_images)