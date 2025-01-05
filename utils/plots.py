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

def plot_feature_map(autoencoder, layer_name, input_image, wandb):
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
    wandb.log({f"Feature map{layer_name}": wandb.Image(plt)})
    plt.show()

from evaluation import get_dist_based_threshold, calculate_error

def plot_images_with_info(autoencoder: Model, test_generator: ImageDataGenerator,threshold_generator: ImageDataGenerator, loss_function: str, n_images: int, title: str, wandb) -> None:
    """
    Plot images from the test generator along with their labels, reconstruction error, and anomaly status.

    Parameters:
    autoencoder (Model): The autoencoder model.
    test_generator (ImageDataGenerator): The test data generator.
    loss_function (str): The loss function to use ('mae', 'mse', 'ssim').
    threshold (float): The threshold for anomaly detection based on reconstruction error.
    n_images (int): The number of images to plot.
    title (str): The title of the plot.
    """
    threshold = get_dist_based_threshold(
        autoencoder=autoencoder,
        threshold_generator=threshold_generator,
        loss_function=loss_function
    )
    test_images, test_labels = next(test_generator)
    reconstructions = autoencoder.predict(test_images, verbose=0)
    errors = calculate_error(test_images, reconstructions, loss_function)

    fig, axes = plt.subplots(n_images, 3, figsize=(15, 5 * n_images))
    fig.suptitle(title, fontsize=16)

    for i in range(n_images):
        is_anomaly = "Yes" if errors[i] > threshold else "No"
        label = "Normal" if test_labels[i][test_generator.class_indices['good']] == 1 else "Anomaly"

        axes[i, 0].imshow(test_images[i])
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"Original (Label: {label})")

        axes[i, 1].imshow(reconstructions[i])
        axes[i, 1].axis("off")
        axes[i, 1].set_title(f"Reconstruction")

        axes[i, 2].text(
            0.5, 0.5, 
            f"Error: {errors[i]:.4f}\nAnomaly: {is_anomaly}",
            fontsize=12, 
            ha="center", 
            va="center"
        )
        axes[i, 2].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    wandb.log({"plot_image_with_info": wandb.Image(plt)})
    plt.show()