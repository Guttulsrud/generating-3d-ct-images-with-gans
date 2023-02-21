import tensorflow as tf
from tensorflow import keras
from config import config


def build_discriminator():
    in_src_image = keras.layers.Input(shape=(*config['images']['shape'], 1))

    discriminator = keras.layers.Conv3D(filters=16,
                                        kernel_size=(3, 3, 3),
                                        strides=(1, 1, 1),
                                        padding='same',
                                        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(in_src_image)
    discriminator = keras.layers.LeakyReLU(alpha=0.2)(discriminator)

    discriminator = keras.layers.Conv3D(filters=32,
                                        kernel_size=(3, 3, 3),
                                        strides=(1, 1, 1),
                                        padding='same',
                                        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(discriminator)
    discriminator = keras.layers.BatchNormalization()(discriminator)
    discriminator = keras.layers.LeakyReLU(alpha=0.2)(discriminator)

    discriminator = keras.layers.Conv3D(filters=64,
                                        kernel_size=(3, 3, 3),
                                        strides=(1, 1, 1),
                                        padding='same',
                                        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(discriminator)
    discriminator = keras.layers.BatchNormalization()(discriminator)
    discriminator = keras.layers.LeakyReLU(alpha=0.2)(discriminator)

    discriminator = keras.layers.Conv3D(filters=128,
                                        kernel_size=(3, 3, 3),
                                        strides=(1, 1, 1),
                                        padding='same',
                                        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(discriminator)
    discriminator = keras.layers.BatchNormalization()(discriminator)
    discriminator = keras.layers.LeakyReLU(alpha=0.2)(discriminator)

    discriminator = keras.layers.Conv3D(filters=1,
                                        kernel_size=(3, 3, 3),
                                        padding='same',
                                        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(discriminator)

    discriminator = keras.layers.Flatten()(discriminator)
    discriminator = keras.layers.Dense(1, activation='sigmoid')(discriminator)

    return keras.models.Model(inputs=in_src_image, outputs=discriminator, name='discriminator')


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss