from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Flatten, Dense, Reshape, Dropout

from utils.loss import return_loss
from tensorflow.keras.applications import MobileNetV2

def get_model(config):
    if config.model_name == "vanilla_autoencoder":
        return vanilla_autoencoder(
            input_shape=(256, 256, 3), 
            optimizer=config.optimizer,
            latent_dim=config.latent_dim, 
            loss=config.loss,
            batch_norm=config.batch_norm,
        )
    elif config.model_name == "deep_autoencoder":
        return deep_autoencoder(
            input_shape=(256, 256, 3), 
            optimizer=config.optimizer,
            latent_dim=config.latent_dim, 
            loss=config.loss,
            batch_norm=config.batch_norm,
            dropout_value=config.dropout_value,
        )
    elif config.model_name == "autoencoder":
        return autoencoder(
            input_shape=(256, 256, 3), 
            optimizer=config.optimizer,
            latent_dim=config.latent_dim, 
            loss=config.loss,
            batch_norm=config.batch_norm,
            dropout_value=config.dropout_value,
            decoder_type=config.decoder_type,
            num_blocks=config.num_blocks
        )
    elif config.model_name == "mobilenet_autoencoder":
        return mobilenet_autoencoder(
            input_shape=(256, 256, 3), 
            optimizer=config.optimizer,
            latent_dim=config.latent_dim, 
            loss=config.loss,
            batch_norm=config.batch_norm,
            decoder_type=config.decoder_type,
            num_blocks=config.num_blocks
        )
    else:
        raise ValueError(f"Model name '{config.model_name}' not recognized.")

def mobilenet_autoencoder(input_shape, optimizer, latent_dim, loss, batch_norm, decoder_type, num_blocks):

    filters = [32 * (2 ** i) for i in range(min(num_blocks, 5))] + [512] * (num_blocks - 5)  # Dynamically set filters based on num_blocks and cap at 512
    
    # Encoder
    input_img = Input(shape=input_shape, name = 'input_layer')
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model(input_img)

    # Bottleneck
    x = Flatten(name='bottleneck_flatten')(x)
    encoded = Dense(latent_dim, name='bottleneck_dense')(x)
    encoded = BatchNormalization(name='bottleneck_batchnorm')(encoded)
    encoded = LeakyReLU(name='bottleneck')(encoded)

    # Decoder
    x = Dense((input_shape[0] // (2 ** num_blocks)) * (input_shape[1] // (2 ** num_blocks)) * filters[-1], name='Dec_Dense')(encoded)
    x = Reshape((input_shape[0] // (2 ** num_blocks), input_shape[1] // (2 ** num_blocks), filters[-1]), name='Dec_Reshape')(x)
    
    for i in reversed(range(num_blocks)):
        if decoder_type == 'upsampling':
            x = UpSampling2D((2, 2), name = f'Dec_UpSampling2D_{num_blocks*2-i}')(x)
            x = Conv2D(filters[i], (3, 3), padding='same', name = f'Dec_Conv2D_{num_blocks*2-i}')(x)
        elif decoder_type == 'transposed':
            x = Conv2DTranspose(filters[i], (3, 3), strides=(2, 2), padding='same', name = f'Dec_ConvTrans_{num_blocks*2-i}')(x)
        x = BatchNormalization(name = f'Dec_BatchNorm_{num_blocks*2-i}')(x) if batch_norm else x
        x = LeakyReLU(name = f'Dec_LeakyReLU_{num_blocks*2-i}')(x)
    
    # Final Output
    x = Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same', name = f'Output_Conv2D')(x)
    
    # Autoencoder Model
    autoencoder = Model(input_img, x)
    autoencoder.compile(optimizer=optimizer, loss=return_loss(loss))
    return autoencoder

def autoencoder(input_shape, optimizer, latent_dim, loss, dropout_value, batch_norm, decoder_type, num_blocks):

    filters = [32 * (2 ** i) for i in range(min(num_blocks, 5))] + [512] * (num_blocks - 5)  # Dynamically set filters based on num_blocks and cap at 512

    # Encoder
    input_img = Input(shape=input_shape, name = 'input_layer')
    x = input_img
    
    for i in range(num_blocks):
        x = Conv2D(filters[i], (3, 3), padding='same', name = f'Enc_Conv2D_{i+1}')(x)
        x = BatchNormalization(name = f'Enc_BathchNorm_{i+1}')(x) if batch_norm else x
        x = LeakyReLU(name = f'Enc_LeakyReLU_{i+1}')(x)
        x = MaxPooling2D((2, 2), padding='same',name = f'Enc_MaxPooling2D_{i+1}')(x)
        x = Dropout(dropout_value, name = f'Enc_Dropout_{i+1}')(x)
    
    # Bottleneck
    x = Flatten(name='bottleneck_flatten')(x)  # Flattened Shape: (32 * 32 * 128,)
    encoded = Dense(latent_dim, name='bottleneck_dense')(x)  # Latent space size reduced to 526
    encoded = BatchNormalization(name='bottleneck_batchnorm')(encoded)
    encoded = LeakyReLU(name='bottleneck')(encoded)
    
    # Decoder
    x = Dense((input_shape[0] // (2 ** num_blocks)) * (input_shape[1] // (2 ** num_blocks)) * filters[-1], name='Dec_Dense')(encoded)
    x = Reshape((input_shape[0] // (2 ** num_blocks), input_shape[1] // (2 ** num_blocks), filters[-1]), name='Dec_Reshape')(x)
    
    for i in reversed(range(num_blocks)):
        if decoder_type == 'upsampling':
            x = UpSampling2D((2, 2), name = f'Dec_UpSampling2D_{num_blocks*2-i}')(x)
            x = Conv2D(filters[i], (3, 3), padding='same', name = f'Dec_Conv2D_{num_blocks*2-i}')(x)
        elif decoder_type == 'transposed':
            x = Conv2DTranspose(filters[i], (3, 3), strides=(2, 2), padding='same', name = f'Dec_ConvTrans_{num_blocks*2-i}')(x)
        x = BatchNormalization(name = f'Dec_BatchNorm_{num_blocks*2-i}')(x) if batch_norm else x
        x = LeakyReLU(name = f'Dec_LeakyReLU_{num_blocks*2-i}')(x)
    
    # Final Output
    x = Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same', name = f'Output_Conv2D')(x)
    
    # Autoencoder Model
    autoencoder = Model(input_img, x)
    autoencoder.compile(optimizer=optimizer, loss=return_loss(loss))
    return autoencoder

def vanilla_autoencoder(input_shape, optimizer, latent_dim, loss, batch_norm):

    # Encoder
    input_img = Input(shape=input_shape)

    # Encoder with reduced complexity
    x = Conv2D(32, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x) if batch_norm else x
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # (128, 128, 32)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x) if batch_norm else x
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # (64, 64, 64)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x) if batch_norm else x
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # (32, 32, 128)
    x = Dropout(0.3)(x)

    # Bottleneck
    x = Flatten()(x)  # Flattened Shape: (32 * 32 * 128,)
    encoded = Dense(latent_dim)(x)  # Latent space size reduced to 526
    encoded = BatchNormalization()(encoded) if batch_norm else encoded
    encoded = LeakyReLU(name='bottleneck')(encoded)

    # Decoder with reduced complexity
    x = Dense(32 * 32 * 128)(encoded)
    x = Reshape((32, 32, 128))(x)

    x = UpSampling2D((2, 2))(x)  # (64, 64, 128)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x) if batch_norm else x
    x = LeakyReLU()(x)

    x = UpSampling2D((2, 2))(x)  # (128, 128, 64)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x) if batch_norm else x
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
    x = BatchNormalization()(x) if batch_norm else x
    x = Dropout(dropout_value)(x)

    x = Conv2D(32, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    x = BatchNormalization()(x) if batch_norm else x
    x = Dropout(dropout_value)(x)

    x = Conv2D(32, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    x = BatchNormalization()(x) if batch_norm else x
    x = Dropout(dropout_value)(x)

    x = Conv2D(64, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    x = BatchNormalization()(x) if batch_norm else x
    x = Dropout(dropout_value)(x)

    x = Conv2D(64, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    x = BatchNormalization()(x) if batch_norm else x
    x = Dropout(dropout_value)(x)

    x = Conv2D(128, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    x = BatchNormalization()(x) if batch_norm else x
    x = Dropout(dropout_value)(x)

    x = Conv2D(64, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    x = BatchNormalization()(x) if batch_norm else x
    x = Dropout(dropout_value)(x)

    x = Conv2D(32, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding="same")(x)
    x = BatchNormalization()(x) if batch_norm else x
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
    autoencoder.compile(optimizer=optimizer, loss=return_loss(loss))
    return autoencoder