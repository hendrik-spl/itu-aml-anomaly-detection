import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def plot_history(comment, history, wandb):
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

    wandb.log({"history_plot": wandb.Image(plt)})

    plt.show()

    print(f'Best train_loss: {np.min(history.history["loss"]).round(4)}')
    print(f'Best val_loss: {np.min(history.history["val_loss"]).round(4)}')
    print(f'Last improvement of val_loss at epoch: {np.argmin(history.history["val_loss"])+1}')

def plot_reconstructions(autoencoder: Model, test_generator: ImageDataGenerator, n_images: int, title, wandb) -> None:
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

    wandb.log({"reconstructions_plot": wandb.Image(plt)})

    plt.show()

def plot_feature_map(autoencoder, layer_name, input_image):
    """
    Visualize the first 10 feature maps for a specific layer of the autoencoder using original colors.
    
    Args:
        autoencoder (Model): The autoencoder model.
        layer_name (str): Name of the layer to visualize feature maps.
        input_image (numpy.ndarray): Input image to generate feature maps.
    """

    # Create a model that outputs feature maps for the specified layer
    layer_output = autoencoder.get_layer(name=layer_name).output
    feature_map_model = Model(inputs=autoencoder.input, outputs=layer_output)

    # Get feature maps
    feature_map = feature_map_model.predict(np.expand_dims(input_image, axis=0))

    # Normalize feature map values to [0, 1] for proper display
    feature_map -= feature_map.min()
    feature_map /= feature_map.max()

    # Limit to the first 10 feature maps
    num_filters = min(1, feature_map.shape[-1])

    # Plot feature maps
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Feature Maps for Layer: {layer_name}", fontsize=12)

    for i in range(num_filters):
        plt.subplot(2, num_filters, i + 1)
        plt.imshow(feature_map[0, :, :, i])  # Display original colors
        plt.axis('off')

    plt.subplots_adjust(top=0.85)  # Add space at the top of the figure
    plt.tight_layout()
    plt.show()