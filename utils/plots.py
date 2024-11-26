import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def plot_history(comment, history):
    """
    Plot training history.
    
    Parameters:
        comment (str): Comment to display in the plot.
        history (tf.keras.callbacks.History): Training history.

    Returns:
        None
    """
    plt.figure(figsize=(14, 4))
    plt.suptitle(comment)

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    print(f'Best train_loss: {np.min(history.history["loss"]).round(4)}')
    print(f'Best val_loss: {np.min(history.history["val_loss"]).round(4)}')
    print(f'Last improvement of val_loss at epoch: {np.argmax(history.history["val_loss"])+1}')

def plot_reconstructions(autoencoder: Model, test_generator: ImageDataGenerator, n_images: int, title) -> None:
    """
    Plot original and reconstructed images from the autoencoder.

    Parameters:
    autoencoder (Model): The autoencoder model.
    test_generator (ImageDataGenerator): The test data generator.
    n_images (int): The number of images to plot.
    """
    test_images, _ = next(test_generator)
    reconstructions = autoencoder.predict(test_images)

    fig, axes = plt.subplots(2, n_images, figsize=(20, 5))
    fig.suptitle(title, fontsize=16)

    for i in range(n_images):
        axes[0, i].imshow(test_images[i])
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')

        axes[1, i].imshow(reconstructions[i])
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstruction')

    plt.show()