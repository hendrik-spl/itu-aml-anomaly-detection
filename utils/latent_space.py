import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

from tensorflow.keras.models import Model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_latent_space(autoencoder, generator, wandb=None, layer_name='bottleneck', n_components=2, generator_type=None):
    """
    Extract latent representations, apply t-SNE, and plot the latent space with class labels or a single color.

    Parameters:
        autoencoder (Model): The full autoencoder model.
        generator (Sequence): The data generator for images.
        layer_name (str): The name of the bottleneck layer to extract features from.
        n_components (int): Number of dimensions for t-SNE.
        generator_type (str): Type of generator ('test', 'train', or 'validation').
        wandb: Weights and Biases logger instance (optional).
    """

    # 🟢 **Step 1: Extract Latent Representations**
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(layer_name).output)
    all_images = []
    all_labels = []

    for i in range(len(generator)):
        batch_images, *batch_labels = next(generator)
        all_images.append(batch_images)
        if generator_type == 'test' and batch_labels:
            all_labels.append(batch_labels[0])  # Only extract labels if available

    # Concatenate batches
    original_images = np.concatenate(all_images)
    latent_representations = encoder.predict(original_images, verbose=0)
    latent_flat = latent_representations.reshape(latent_representations.shape[0], -1)

    # 🟡 **Step 2: Handle Labels**
    if generator_type == 'test':
        labels = np.concatenate(all_labels)
        if 'good' in generator.class_indices:
            relevant_label_index = generator.class_indices["good"]
            labels_indices = (labels[:, relevant_label_index] == 0).astype(int)  # Binary: 0 for good, 1 for anomaly
        else:
            raise ValueError("Ensure that the generator contains a 'good' class for normal samples.")
    elif generator_type == 'train':
        # Training data has no labels → Assign a uniform label
        labels_indices = np.zeros(latent_flat.shape[0])  # All points labeled as 'Train'
    elif generator_type == 'validation':
        labels_indices = np.zeros(latent_flat.shape[0])  # All points labeled as 'Validation'
    else:
        raise ValueError("generator_type must be 'test', 'train', or 'validation'.")

    # 🟣 **Step 3: Apply t-SNE for Dimensionality Reduction**
    tsne = TSNE(n_components=n_components, random_state=42)
    latent_2d = tsne.fit_transform(latent_flat)

    # 🟠 **Step 4: Plot Latent Space**
    plt.figure(figsize=(10, 7))
    if generator_type == 'test':
        # Plot with class labels
        class_indices = {'good': 0, 'anomaly': 1}
        index_to_class = {v: k for k, v in class_indices.items()}
        cmap = plt.cm.get_cmap('viridis', len(class_indices))
        patches = [mpatches.Patch(color=cmap(i), label=index_to_class[i]) for i in range(len(class_indices))]

        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels_indices, cmap=cmap, alpha=0.7)
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    elif generator_type == 'train':
        # Plot all training data points in one color
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c='blue', alpha=0.7, label='Train Data')
        plt.legend()

    elif generator_type == 'validation':
        # Plot all validation data points in one color
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c='green', alpha=0.7, label='Validation Data')
        plt.legend()

    plt.title(f'Latent Space Visualization ({generator_type.capitalize()} Data)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()

    if wandb:
        wandb.log({f"{generator_type}_latent_space_plot": wandb.Image(plt)})
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



from tensorflow.keras.models import Model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_combined_latent_space(autoencoder, train_generator, validation_generator, test_generator, 
                               wandb=None, layer_name='bottleneck', n_components=2):
    """
    Extract latent representations from train, validation, and test datasets, apply t-SNE, and plot the combined latent space.
    Test data retains separate labels for 'good' and 'anomaly'.

    Parameters:
        autoencoder (Model): The full autoencoder model.
        train_generator (Sequence): The data generator for training images.
        validation_generator (Sequence): The data generator for validation images.
        test_generator (Sequence): The data generator for test images.
        layer_name (str): The name of the bottleneck layer to extract features from.
        n_components (int): Number of dimensions for t-SNE projection.
        wandb: Weights and Biases logger instance (optional).
    """
    #Step 1: Extract Latent Representations from Generators**
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(layer_name).output)
    
    #Extract Train Data
    train_images = []
    for i in range(len(train_generator)):
        batch_images, *_ = next(train_generator)
        train_images.append(batch_images)
    train_images = np.concatenate(train_images)
    train_latent = encoder.predict(train_images, verbose=0)
    train_latent_flat = train_latent.reshape(train_latent.shape[0], -1)
    
    #Extract Validation Data
    validation_images = []
    for i in range(len(validation_generator)):
        batch_images, *_ = next(validation_generator)
        validation_images.append(batch_images)
    validation_images = np.concatenate(validation_images)
    validation_latent = encoder.predict(validation_images, verbose=0)
    validation_latent_flat = validation_latent.reshape(validation_latent.shape[0], -1)
    
    #Extract Test Data with Labels
    test_images = []
    test_labels = []
    for i in range(len(test_generator)):
        batch_images, batch_labels = next(test_generator)
        test_images.append(batch_images)
        test_labels.append(batch_labels)
    test_images = np.concatenate(test_images)
    test_labels = np.concatenate(test_labels)
    test_latent = encoder.predict(test_images, verbose=0)
    test_latent_flat = test_latent.reshape(test_latent.shape[0], -1)
    
    #Step 2: Map Test Labels to 'good' and 'anomaly'**
    if 'good' in test_generator.class_indices:
        good_label_index = test_generator.class_indices["good"]
        test_labels_indices = (test_labels[:, good_label_index] == 0).astype(int)  # 0: good, 1: anomaly
    else:
        raise ValueError("Ensure that the test generator contains a 'good' class for normal samples.")
    
    #Step 3: Combine Data and Labels**
    combined_latent = np.vstack([train_latent_flat, validation_latent_flat, test_latent_flat])
    combined_labels = np.concatenate([
        np.full(len(train_latent_flat), fill_value=0),  # Label 0 for Train Data
        np.full(len(validation_latent_flat), fill_value=1),  # Label 1 for Validation Data
        test_labels_indices + 2  # 2 for Good (from test), 3 for Anomaly (from test)
    ])
    
    #Step 4: Apply t-SNE for Dimensionality Reduction**
    tsne = TSNE(n_components=n_components, random_state=42)
    latent_2d = tsne.fit_transform(combined_latent)
    
    #Step 5: Plot the Combined Latent Space**
    plt.figure(figsize=(12, 8))
    
    # Plot Train Data
    plt.scatter(
        latent_2d[combined_labels == 0, 0],
        latent_2d[combined_labels == 0, 1],
        alpha=0.7,
        label='Train Data',
        color='blue'
    )
    
    # Plot Validation Data
    plt.scatter(
        latent_2d[combined_labels == 1, 0],
        latent_2d[combined_labels == 1, 1],
        alpha=0.7,
        label='Validation Data',
        color='cyan'
    )
    
    # Plot Test Data - Good
    plt.scatter(
        latent_2d[combined_labels == 2, 0],
        latent_2d[combined_labels == 2, 1],
        alpha=0.7,
        label='Test Data (Good)',
        color='orange'
    )
    
    # Plot Test Data - Anomaly
    plt.scatter(
        latent_2d[combined_labels == 3, 0],
        latent_2d[combined_labels == 3, 1],
        alpha=0.7,
        label='Test Data (Anomaly)',
        color='red'
    )
    
    plt.title('Combined Latent Space Visualization (Train + Validation + Test - Good/Anomaly)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.tight_layout()
    
    # Log to Weights and Biases if enabled
    if wandb:
        wandb.log({"combined_latent_space_plot": wandb.Image(plt)})
    
    plt.show()


