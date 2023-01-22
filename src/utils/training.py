import tensorflow as tf
import os


# The training loop begins with generator receiving a random seed as input. That seed is used to produce an image.
# The discriminator is then used to classify real images (drawn from the training set) and
# fakes images (produced by the generator).
# The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.

# This annotation causes the function to be "compiled".
@tf.function
def train_step(generator,
               discriminator,
               generator_loss,
               discriminator_loss,
               generator_optimizer,
               discriminator_optimizer, images):
    noise = tf.random.normal([1, 16, 16, 32])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def create_checkpoint(generator, discriminator, generator_optimizer, discriminator_optimizer):
    checkpoint_dir = '../training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    return checkpoint, checkpoint_prefix
