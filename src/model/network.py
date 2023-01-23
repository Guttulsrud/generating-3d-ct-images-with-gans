from matplotlib import pyplot as plt

from src.model.discriminator import build_discriminator, discriminator_loss
from src.model.generator import build_generator, generator_loss
import tensorflow as tf
import os


class Network:
    def __init__(self, load_checkpoint=False):

        if load_checkpoint:
            self.restore_checkpoint()
            return
        image_shape = (16, 16, 32)
        self.seed = tf.random.normal([1, 16, 16, 32])
        self.generator = build_generator(input_shape=(*image_shape, 1))
        self.discriminator = build_discriminator(input_shape=(*image_shape, 1))
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.checkpoint_prefix = os.path.join('../training_checkpoints', "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    @tf.function
    def train(self, images):
        # The training loop begins with generator receiving a random seed as input. That seed is used to produce an
        # image. The discriminator is then used to classify real images (drawn from the training set) and fakes
        # images (produced by the generator). The loss is calculated for each of these models, and the gradients are
        # used to update the generator and discriminator.

        # This annotation causes the function to be "compiled".
        noise = tf.random.normal([1, 16, 16, 32])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def save_checkpoint(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def restore_checkpoint(self):
        self.checkpoint.restore(tf.train.latest_checkpoint('../training_checkpoints'))

    def save_images(self, epoch):
        generated_images = self.generator(self.seed, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(generated_images.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()