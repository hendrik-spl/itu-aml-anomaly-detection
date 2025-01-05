import keras
import tensorflow as tf
import numpy as np

@keras.saving.register_keras_serializable()
def ssim_loss(y_true, y_pred):
    # https://medium.com/@majpaw1996/anomaly-detection-in-computer-vision-with-ssim-ae-2d5256ffc06b
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def return_loss(loss):
    if loss == 'mse':
        return 'mean_squared_error'
    elif loss == 'mae':
        return 'mean_absolute_error'
    elif loss == 'ssim':
        return ssim_loss
    else:
        raise ValueError(f"Unknown loss function: {loss}")

def calculate_error(images: np.ndarray, reconstructions: np.ndarray, loss_function: str) -> List[float]:
    """
    Calculate error between original images and their reconstructions.

    Parameters:
    images (np.ndarray): The original images.
    reconstructions (np.ndarray): The reconstructed images.
    loss_function (str): The loss function to use ('mae', 'mse', 'ssim').

    Returns:
    List[float]: A list of errors for each image in the batch.
    """
    if loss_function == 'mae':
        return np.mean(np.abs(reconstructions - images), axis=(1, 2, 3)).tolist()
    elif loss_function == 'mse':
        return np.mean(np.square(reconstructions - images), axis=(1, 2, 3)).tolist()
    elif loss_function == 'ssim':
        return ssim_loss(images, reconstructions).numpy().tolist()
    else:
        raise ValueError(f"Unknown loss function: {loss_function}. Please define a function to calculate the error.")