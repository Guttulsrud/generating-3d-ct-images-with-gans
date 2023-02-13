from matplotlib import pyplot as plt
from config import config
from src.model.discriminator import build_discriminator, discriminator_loss
from src.model.generator import build_generator, generator_loss
import tensorflow as tf
import os


class Network:
    def __init__(self, start_datetime, load_checkpoint=False):
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

        self.checkpoint_prefix = os.path.join(f'logs/{self.start_datetime}/training_checkpoints', "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    # This annotation causes the function to be "compiled" with TF.
    @tf.function
    def train(self, images):
        # The training loop begins with generator receiving a random seed as input. That seed is used to produce an
        # image. The discriminator is then used to classify real images (drawn from the training set) and fakes
        # images (produced by the generator). The loss is calculated for each of these models, and the gradients are
        # used to update the generator and discriminator.

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
        self.checkpoint.restore(tf.train.latest_checkpoint(f'logs/{self.start_datetime}/training_checkpoints'))

    def save_images(self, epoch):
        generated_images = self.generator(self.seed, training=False)
        for i in range(generated_images.shape[0]):
            plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(f'../logs/{self.start_datetime}/epoch_images/Epoch_{epoch}.png')
        plt.show()
