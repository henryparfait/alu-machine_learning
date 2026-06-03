#!/usr/bin/env python3
"""Creates a variational autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Returns the encoder, decoder, and full variational autoencoder."""
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)
    mean = keras.layers.Dense(latent_dims)(encoded)
    log_var = keras.layers.Dense(latent_dims)(encoded)

    def sampling(args):
        """Reparameterization trick: z = mean + exp(log_var / 2) * eps."""
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        eps = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * eps

    z = keras.layers.Lambda(sampling)([mean, log_var])
    encoder = keras.Model(inputs, [z, mean, log_var])

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    decoded = latent_inputs
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(latent_inputs, outputs)

    # Full autoencoder (decoder takes the sampled z)
    auto = keras.Model(inputs, decoder(encoder(inputs)[0]))

    def vae_loss(y_true, y_pred):
        """Reconstruction (BCE) + KL divergence."""
        reconstruction = keras.backend.binary_crossentropy(y_true, y_pred)
        reconstruction = keras.backend.sum(reconstruction, axis=1)
        kl = 1 + log_var - keras.backend.square(mean) - \
            keras.backend.exp(log_var)
        kl = -0.5 * keras.backend.sum(kl, axis=1)
        return reconstruction + kl

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
