from model.GAN import GAN
from model.discriminator import build_discriminator
from model.generator import build_generator
import tensorflow as tf
import os
from model.loss_functions import generator_loss, discriminator_loss


class VanillaGAN(GAN):
    def __init__(self, start_datetime, config):
        super().__init__(start_datetime, config)

        self.generator_optimizer = tf.keras.optimizers.Adam(self.generator_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.discriminator_learning_rate)
        self.init_checkpoint()

    @tf.function
    def train(self, images, epoch):
        self.epoch = int(epoch)

        noise = tf.random.normal([1, *self.config['images']['shape']])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Calculate loss for generator and discriminator
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

            self.log_scalars(gen_loss, disc_loss)
            self.log_images(generated_images)

        # Get the gradients for each model
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Combine gradients with training variables
        generator_gradients = zip(gradients_of_generator, self.generator.trainable_variables)
        discriminator_gradients = zip(gradients_of_discriminator, self.discriminator.trainable_variables)

        # Apply gradients to the models
        self.generator_optimizer.apply_gradients(generator_gradients)
        self.discriminator_optimizer.apply_gradients(discriminator_gradients)

        self.log_scalars(gen_loss, disc_loss)
        self.log_images(generated_images)

