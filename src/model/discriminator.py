from tensorflow import keras


def define_discriminator(image_shape):
    in_src_image = keras.layers.Input(shape=image_shape)
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


# d = define_discriminator((16, 16, 32,))
# d.summary()
