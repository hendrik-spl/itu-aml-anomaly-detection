import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

def plot_latent_space(autoencoder, test_generator, wandb, layer_name='bottleneck', n_components=2):
    """
    Extract latent representations, apply t-SNE, and plot the latent space with class labels.

    Parameters:
    autoencoder (Model): The full autoencoder model.
    test_generator (Sequence): The data generator for test images.
    layer_name (str): The name of the bottleneck layer to extract features from.
    n_components (int): Number of dimensions for t-SNE.
    """

    # Extract latent representations
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(layer_name).output)
    all_images, all_labels = zip(*(next(test_generator) for _ in range(len(test_generator))))

    # Concatenate all batches
    original_images = np.concatenate(all_images)
    labels = np.concatenate(all_labels)

    # Get the relevant class index programmatically
    relevant_label_index = test_generator.class_indices["good"]

    # Convert one-hot encoded labels to binary class indices
    labels_indices = labels[:, relevant_label_index]

    # Get latent representations and flatten spatial dimensions
    latent_representations = encoder.predict(original_images, verbose=0)
    latent_flat = latent_representations.reshape(latent_representations.shape[0], -1)

    # Step 2: Apply t-SNE
    tsne = TSNE(n_components=n_components, random_state=42)
    latent_2d = tsne.fit_transform(latent_flat)

    # Step 3: Plot the latent space
    class_indices = {'good': 0, 'anomaly': 1}
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
    if wandb: wandb.log({"latent_space_plot" : wandb.Image(plt)})
    plt.show()

def plot_3d_latent_space(autoencoder, test_generator, layer_name='bottleneck', n_components=3, angles=None, wandb=None):
    """
    Extract latent representations, apply t-SNE, and plot the 3D latent space with class labels from multiple angles.

    Parameters:
    autoencoder (Model): The full autoencoder model.
    test_generator (Sequence): The data generator for test images.
    layer_name (str): The name of the bottleneck layer to extract features from.
    n_components (int): Number of dimensions for t-SNE (default: 3).
    angles (list of tuples): List of (elevation, azimuth) angles to view the plot.
    """

    # Default angles if none provided
    if angles is None:
        angles = [(30, 45), (30, 135), (60, 45)]

    # Extract latent representations
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

    # Apply t-SNE for 3D transformation
    tsne = TSNE(n_components=n_components, random_state=42)
    latent_3d = tsne.fit_transform(latent_flat)

    # Define class mappings
    class_indices = {i: f'Class {i}' for i in np.unique(labels_indices)}
    num_classes = len(class_indices)

    # Define colormap and legend patches
    cmap = plt.cm.get_cmap('viridis', num_classes)
    patches = [mpatches.Patch(color=cmap(i), label=f'Class {i}') for i in range(num_classes)]

    # Create 3D plots for each angle
    for angle in angles:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for latent representations
        scatter = ax.scatter(latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2],
                             c=labels_indices, cmap=cmap, alpha=0.7)

        # Set plot properties
        ax.set_title(f'3D Latent Space Visualization\n(Elev: {angle[0]}, Azim: {angle[1]})')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_zlabel('t-SNE Dimension 3')

        # Adjust view angle
        ax.view_init(elev=angle[0], azim=angle[1])

        # Add legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        if wandb: wandb.log({f"latent_3d_space_plot{angle}" : wandb.Image(plt)})
        plt.show()