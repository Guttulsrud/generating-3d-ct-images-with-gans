from tensorflow import keras


# define an encoder block
def encoder(layer_in, n_filters, batch_normalization=True):
    # weight initialization
    init = keras.initializers.RandomNormal(stddev=0.02)
    # add down sampling layer
    g = keras.layers.Conv3D(filters=n_filters,
                            kernel_size=(3, 3, 3),
                            strides=(1, 1, 1),
                            padding='same',
                            kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batch_normalization:
        g = keras.layers.BatchNormalization()(g, training=True)

    # leaky relu activation
    return keras.layers.LeakyReLU(alpha=0.2)(g)


# define a decoder block
def decoder(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = keras.initializers.RandomNormal(stddev=0.02)
    # add up sampling layer
    g = keras.layers.Conv3DTranspose(filters=n_filters,
                                     kernel_size=(3, 3, 3),
                                     strides=(1, 1, 1),
                                     padding='same',
                                     kernel_initializer=init)(layer_in)
    # add batch normalization
    g = keras.layers.BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = keras.layers.Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = keras.layers.Concatenate()([g, skip_in])
    # relu activation

    return keras.layers.Activation('relu')(g)


def build_generator(config):
    in_image = keras.layers.Input(shape=(*config['images']['shape'], 1))

    # Encoder
    encoder_1 = encoder(in_image, 16, batch_normalization=False)
    encoder_2 = encoder(encoder_1, 32)
    encoder_3 = encoder(encoder_2, 64)
    encoder_4 = encoder(encoder_3, 128)

    # Bottleneck
    bottleneck = keras.layers.Conv3D(filters=128,
                                     kernel_size=(3, 3, 3),
                                     strides=(1, 1, 1),
                                     padding='same',
                                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(encoder_4)
    bottleneck = keras.layers.Activation('relu')(bottleneck)

    # Decoder
    decoder_1 = decoder(bottleneck, encoder_4, 128, dropout=False)
    decoder_2 = decoder(decoder_1, encoder_3, 64, dropout=False)
    decoder_3 = decoder(decoder_2, encoder_2, 32, dropout=False)
    decoder_4 = decoder(decoder_3, encoder_1, 16, dropout=False)

    # output
    output = keras.layers.Conv3DTranspose(filters=1,
                                          kernel_size=(3, 3, 3),
                                          strides=(1, 1, 1),
                                          padding='same',
                                          kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                                          activation='tanh')(decoder_4)
    model = keras.models.Model(inputs=in_image, outputs=output, name='generator')
    # print(model.output_shape)
    return model
