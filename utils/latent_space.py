import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model

def extract_latent_representations(autoencoder, test_generator, layer_name='bottleneck'):
    """
    Extract latent representations from the encoder part of an autoencoder.
    
    Parameters:
    autoencoder (Model): The full autoencoder model.
    test_generator (Sequence): The data generator for test images.
    layer_name (str): The name of the bottleneck layer to extract features from.

    Returns:
    np.ndarray: Flattened latent representations of the images.
    np.ndarray: Original image labels.
    """
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(layer_name).output)

    all_images, all_labels = zip(*(next(test_generator) for _ in range(len(test_generator))))

    # Concatenate all batches
    original_images = np.concatenate(all_images)
    labels = np.concatenate(all_labels)

    # Convert one-hot encoded labels to binary class indices
    labels_indices = np.where((labels == [1, 0, 0, 0, 0, 0]).all(axis=1), 0, 1)

    # Get latent representations and flatten spatial dimensions
    latent_representations = encoder.predict(original_images, verbose=0)
    latent_flat = latent_representations.reshape(latent_representations.shape[0], -1)

    return latent_flat, labels_indices

def apply_tsne(latent_representations, n_components=2):
    """
    Apply t-SNE to reduce dimensionality of latent representations.

    Parameters:
    latent_representations (np.ndarray): Flattened latent representations of images.
    n_components (int): Number of dimensions for t-SNE.
    perplexity (float): Perplexity parameter for t-SNE.
    n_iter (int): Number of iterations for optimization.

    Returns:
    np.ndarray: 2D representations of the latent space.
    """
    tsne = TSNE(n_components=n_components)
    latent_2d = tsne.fit_transform(latent_representations)
    return latent_2d

def plot_latent_space(latent_2d, labels_indices, class_indices):
    """
    Plot the latent space with class labels.

    Parameters:
    latent_2d (np.ndarray): 2D representations of the latent space.
    labels_indices (np.ndarray): Class indices for each image.
    class_indices (dict): Mapping of class names to class indices.
    """
    index_to_class = {v: k for k, v in class_indices.items()}
    num_classes = len(class_indices)

    # Define colormap and legend patches
    cmap = plt.cm.get_cmap('viridis', num_classes)
    patches = [mpatches.Patch(color=cmap(i), label=index_to_class[i]) for i in range(num_classes)]

    # Plot latent space with class labels
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels_indices, cmap=cmap, alpha=0.7)
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Latent Space Visualization with Hard Class Labels')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.show()