import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns

from utils.evaluation import get_dist_based_threshold, calculate_error

def plot_history(comment, history, wandb = None):
    plt.figure(figsize=(14, 4))
    plt.suptitle(comment)

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if wandb: wandb.log({"history_plot": wandb.Image(plt)})

    plt.show()

    print(f'Best train_loss: {np.min(history.history["loss"]).round(4)}')
    print(f'Best val_loss: {np.min(history.history["val_loss"]).round(4)}')
    print(f'Last improvement of val_loss at epoch: {np.argmin(history.history["val_loss"])+1}')
    if wandb: wandb.log({"last_improvement_epoch": np.argmin(history.history["val_loss"])+1})

def plot_reconstructions(autoencoder: Model, test_generator: ImageDataGenerator, n_images: int, title, wandb = None) -> None:
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

    if wandb: wandb.log({"reconstructions_plot": wandb.Image(plt)})

    plt.show()


def plot_feature_maps(autoencoder, generator, img_index=0, feature_map_index=0, wandb=None):

    conv_layers = [layer.name for layer in autoencoder.layers if 'Conv2D' in layer.name]
    if not conv_layers:
        raise ValueError("No layers containing 'Conv2D' found in the model.")
    
    print(f"Found Conv2D layers: {conv_layers}")

    batch_size = generator.batch_size
    batch_idx = img_index // batch_size
    img_idx_within_batch = img_index % batch_size

    for i, (images, _) in enumerate(generator):
        if i == batch_idx:
            sample_image = images[img_idx_within_batch]
            break
    else:
        raise ValueError("Image index is out of range of the generator.")

    # Loop through each Conv2D layer and plot one feature map
    for layer_name in conv_layers:
        layer_output = autoencoder.get_layer(name=layer_name).output
        feature_map_model = Model(inputs=autoencoder.input, outputs=layer_output)

        feature_map = feature_map_model.predict(np.expand_dims(sample_image, axis=0))

        if feature_map_index >= feature_map.shape[-1]:
            print(f"Skipping {layer_name}: feature_map_index {feature_map_index} exceeds available maps ({feature_map.shape[-1]}).")
            continue

        selected_map = feature_map[0, :, :, feature_map_index]
        selected_map -= selected_map.min()
        selected_map /= selected_map.max()

        plt.figure(figsize=(5, 5))
        plt.title(f"Feature Map {feature_map_index} from Layer: {layer_name}")
        plt.imshow(selected_map, cmap='viridis')
        plt.axis('off')
        plt.tight_layout()

        if wandb: 
            wandb.log({f"Feature Map {layer_name}_{feature_map_index}": wandb.Image(plt)})
        
        plt.show()


def plot_images_with_info(autoencoder: Model, test_generator: ImageDataGenerator,threshold_generator: ImageDataGenerator, loss_function: str, n_images: int, title: str, wandb = None) -> None:

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
    if wandb: wandb.log({"plot_image_with_info": wandb.Image(plt)})
    plt.show()

def plot_single_histogram_with_threshold(errors, threshold: float, title: str, xlabel: str, ylabel: str, threshold_label: str) -> None:
    plt.hist(errors, bins=50, alpha=0.5)
    plt.axvline(threshold, color='r', linestyle='--', label=threshold_label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()