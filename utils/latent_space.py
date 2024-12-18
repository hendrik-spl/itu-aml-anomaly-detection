import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model

def plot_latent_space(autoencoder, test_generator, wandb, layer_name='bottleneck', n_components=2):
    """
    Extract latent representations, apply t-SNE, and plot the latent space with class labels.

    Parameters:
    autoencoder (Model): The full autoencoder model.
    test_generator (Sequence): The data generator for test images.
    layer_name (str): The name of the bottleneck layer to extract features from.
    n_components (int): Number of dimensions for t-SNE.
    """
    from sklearn.manifold import TSNE
    from tensorflow.keras.models import Model
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

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
    wandb.log({"latent_space_plot" : wandb.Image(plt)})
    plt.show()