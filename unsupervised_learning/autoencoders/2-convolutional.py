#!/usr/bin/env python3
"""Creates a convolutional autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Returns the encoder, decoder, and full convolutional autoencoder."""
    # Encoder
    inputs = keras.Input(shape=input_dims)
    encoded = inputs
    for f in filters:
        encoded = keras.layers.Conv2D(f, (3, 3), padding='same',
                                      activation='relu')(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    encoder = keras.Model(inputs, encoded)

    # Decoder
    latent_inputs = keras.Input(shape=latent_dims)
    decoded = latent_inputs
    for f in reversed(filters[1:]):
        decoded = keras.layers.Conv2D(f, (3, 3), padding='same',
                                      activation='relu')(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    # Second-to-last convolution: valid padding
    decoded = keras.layers.Conv2D(filters[0], (3, 3), padding='valid',
                                  activation='relu')(decoded)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    # Last convolution: channels match input, sigmoid, no upsampling
    outputs = keras.layers.Conv2D(input_dims[-1], (3, 3), padding='same',
                                  activation='sigmoid')(decoded)
    decoder = keras.Model(latent_inputs, outputs)

    # Full autoencoder
    auto = keras.Model(inputs, decoder(encoder(inputs)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
