import tensorflow as tf
from tensorflow import keras


def build_discriminator(input_shape):
    in_src_image = keras.layers.Input(shape=input_shape)
    discriminator = keras.layers.Conv3D(filters=64,
                                        kernel_size=(4, 4, 4),
                                        strides=(1, 1, 1),
                                        padding='same',
                                        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(in_src_image)
    discriminator = keras.layers.LeakyReLU(alpha=0.2)(discriminator)

    discriminator = keras.layers.Conv3D(filters=128,
                                        kernel_size=(4, 4, 4),
                                        strides=(1, 1, 1),
                                        padding='same',
                                        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(discriminator)
    discriminator = keras.layers.BatchNormalization()(discriminator)
    discriminator = keras.layers.LeakyReLU(alpha=0.2)(discriminator)

    discriminator = keras.layers.Conv3D(filters=256,
                                        kernel_size=(4, 4, 4),
                                        strides=(1, 1, 1),
                                        padding='same',
                                        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(discriminator)
    discriminator = keras.layers.BatchNormalization()(discriminator)
    discriminator = keras.layers.LeakyReLU(alpha=0.2)(discriminator)

    discriminator = keras.layers.Conv3D(filters=512,
                                        kernel_size=(4, 4, 4),
                                        strides=(1, 1, 1),
                                        padding='same',
                                        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(discriminator)
    discriminator = keras.layers.BatchNormalization()(discriminator)
    discriminator = keras.layers.LeakyReLU(alpha=0.2)(discriminator)

    discriminator = keras.layers.Conv3D(filters=1,
                                        kernel_size=(4, 4, 4),
                                        padding='same',
                                        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(discriminator)

    discriminator = keras.layers.Flatten()(discriminator)
    discriminator = keras.layers.Dense(1, activation='sigmoid')(discriminator)

    return keras.models.Model(inputs=in_src_image, outputs=discriminator, name='discriminator')


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
