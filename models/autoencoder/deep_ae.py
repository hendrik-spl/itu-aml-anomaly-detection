### Shall Contain all Code for models

import keras
import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    LeakyReLU,
    BatchNormalization,
    Flatten,
    Dense,
    Reshape,
    Dropout,
    MaxPooling2D,
    UpSampling2D
)
from tensorflow.keras.models import Model



def deep_autoencoder(input_shape=(256, 256, 3), optimizer = 'adam', flc=32, latent_dim=512,batch_norm=True, dropout=True, dropout_value=0.5, loss='mse'):
    """
    input_shape (tuple): Shape of the input images (height, width, channels).
    flc (int): Number of filters in the initial convolutional layer.
    z_dim (int): Latent space dimension.
    batch_norm (bool): Activate batch normalisation layer
    dropout (bool): activate dropout layer
    dropout_value (float): define own dropout value
    loss (str): define loss used to train autoencoder. Has to be 'mse','msa',ssim_loss,ssim_l1_loss
    """
    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(input_img)
    if batch_norm: x = BatchNormalization()(x)
    if dropout: x = Dropout(dropout_value)(x)
    x = Conv2D(flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(x)
    if batch_norm: x = BatchNormalization()(x)
    if dropout: x = Dropout(dropout_value)(x)
    x = Conv2D(flc, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(x)
    if batch_norm: x = BatchNormalization()(x)
    if dropout: x = Dropout(dropout_value)(x)
    x = Conv2D(flc * 2, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(x)
    if batch_norm: x = BatchNormalization()(x)
    if dropout: x = Dropout(dropout_value)(x)
    x = Conv2D(flc * 2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(x)
    if batch_norm: x = BatchNormalization()(x)
    if dropout: x = Dropout(dropout_value)(x)
    x = Conv2D(flc * 4, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(x)
    if batch_norm: x = BatchNormalization()(x)
    if dropout: x = Dropout(dropout_value)(x)
    x = Conv2D(flc * 2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(x)
    if batch_norm: x = BatchNormalization()(x)
    if dropout: x = Dropout(dropout_value)(x)
    x = Conv2D(flc, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(x)
    if batch_norm: x = BatchNormalization()(x)
    if dropout: x = Dropout(dropout_value)(x)
    encoded = Conv2D(latent_dim, (8, 8), strides=1, activation='linear', padding='valid')(x)

    # Decoder
    x = Conv2DTranspose(flc, (8, 8), strides=1, activation=LeakyReLU(alpha=0.2), padding='valid')(encoded)
    x = Conv2D(flc * 2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(x)
    x = Conv2D(flc * 4, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(x)
    x = Conv2DTranspose(flc * 2, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(x)
    x = Conv2D(flc * 2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(x)
    x = Conv2DTranspose(flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(x)
    x = Conv2D(flc, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(x)
    x = Conv2DTranspose(flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(x)
    decoded = Conv2DTranspose(input_shape[2], (4, 4), strides=2, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return autoencoder
