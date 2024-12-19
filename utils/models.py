from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Flatten, Dense, Reshape, Dropout

from utils.loss import return_loss

def vanilla_autoencoder(input_shape, optimizer, latent_dim, loss):
    """
        input_shape (tuple): Shape of the input images (height, width, channels).
        latent_dim (int): Dimension of the latent space.
        loss (str): Loss used to train the autoencoder. Options: 'mse', 'ssim', etc.
    """
    # Encoder
    input_img = Input(shape=input_shape)

    # Encoder with reduced complexity
    x = Conv2D(32, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # (128, 128, 32)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # (64, 64, 64)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # (32, 32, 128)
    x = Dropout(0.3)(x)

    # Bottleneck
    x = Flatten()(x)  # Flattened Shape: (32 * 32 * 128,)
    encoded = Dense(latent_dim)(x)  # Latent space size reduced to 526
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU(name='bottleneck')(encoded)

    # Decoder with reduced complexity
    x = Dense(32 * 32 * 128)(encoded)
    x = Reshape((32, 32, 128))(x)

    x = UpSampling2D((2, 2))(x)  # (64, 64, 128)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = UpSampling2D((2, 2))(x)  # (128, 128, 64)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = UpSampling2D((2, 2))(x)  # (256, 256, 32)
    x = Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')(x)

    # Autoencoder Model
    autoencoder = Model(input_img, x)
    autoencoder.compile(optimizer=optimizer, loss=return_loss(loss))
    return autoencoder

def deep_autoencoder(input_shape=(256, 256, 3), optimizer="adam",latent_dim=512, loss="mse",batch_norm=True,dropout_value=0.5,):
    """
        Architecture similar to the one used by:
        https://github.com/plutoyuxie/AutoEncoder-SSIM-for-unsupervised-anomaly-detection-/blob/master/train.py
        Added customizable dropout and batchnorm  

    """
    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding="same")(input_img)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dropout(dropout_value)(x)

    x = Conv2D(32, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dropout(dropout_value)(x)

    x = Conv2D(32, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dropout(dropout_value)(x)

    x = Conv2D(64, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dropout(dropout_value)(x)

    x = Conv2D(64, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dropout(dropout_value)(x)

    x = Conv2D(128, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dropout(dropout_value)(x)

    x = Conv2D(64, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dropout(dropout_value)(x)

    x = Conv2D(32, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dropout(dropout_value)(x)

    encoded = Conv2D(latent_dim, (8, 8), strides=1, activation="linear", padding="valid")(x)

    # Decoder
    x = Conv2DTranspose(32, (8, 8), strides=1, activation=LeakyReLU(alpha=0.2), padding="valid")(encoded)
    x = Conv2D(64, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    x = Conv2D(128, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    x = Conv2DTranspose(64, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    x = Conv2D(64, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    x = Conv2DTranspose(32, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    x = Conv2D(32, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    x = Conv2DTranspose(32, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    
    decoded = Conv2DTranspose(input_shape[2], (4, 4), strides=2, activation="sigmoid", padding="same")(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder