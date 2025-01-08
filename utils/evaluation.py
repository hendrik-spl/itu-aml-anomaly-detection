from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    f1_score, 
    roc_auc_score, 
    precision_recall_curve, 
    auc, 
    accuracy_score, 
    roc_curve
)
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from typing import Literal

from utils.loss import calculate_error
from utils.plots_helper import plot_double_histogram_with_threshold, plot_confusion_matrix, plot_roc_curve, plot_smooth_error_distribution

ground_truth_labels = [
    "Normal", # 0
    "Anomaly" # 1
    ]

def extract_errors_and_labels_from_generator(autoencoder: Model, generator: ImageDataGenerator, loss_function: str, relevant_label_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper that goes through all batches in `generator` and accumulates errors & labels.
    This approach ensures we iterate exactly over the number of steps in the generator.
    """
    all_errors = []
    all_labels = []
    
    generator.reset()

    # Typically, len(generator) is the number of batches per epoch
    for _ in range(len(generator)):
        batch_images, batch_labels = next(generator)
        reconstructions = autoencoder.predict(batch_images, verbose=0)
        batch_errors = calculate_error(batch_images, reconstructions, loss_function)
        
        # The relevant_label_index gives us "is good?" for each image. 
        # If batch_labels[:, relevant_label_index] == 1, that means it's 'good'.
        all_errors.extend(batch_errors)
        all_labels.extend(batch_labels[:, relevant_label_index])
    
    # Return as numpy arrays
    return np.array(all_errors), np.array(all_labels)

# Function to calculate errors and labels
def get_errors_and_labels(autoencoder: Model, generator: ImageDataGenerator, loss_function: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate errors and true labels.

    Parameters:
    autoencoder (Model): The autoencoder model.
    generator (ImageDataGenerator): The test data generator.
    loss_function (str): The loss function to use ('mae' or 'mse').

    Returns:
    Tuple[np.ndarray, np.ndarray]: Arrays of test errors and true labels.
    """
    relevant_label_index = generator.class_indices["good"] # The index of the relevant label

    errors, labels_good = extract_errors_and_labels_from_generator(
        autoencoder=autoencoder,
        generator=generator,
        loss_function=loss_function,
        relevant_label_index=relevant_label_index
    )
    # labels_good is 1 if the sample is "good", 0 otherwise.
    # We want final labels to be: 0 => Normal, 1 => Anomaly.
    # If labels_good == 1 => normal => final label 0.
    # If labels_good == 0 => anomaly => final label 1.
    labels = 1 - labels_good

    return errors, labels

# Function to get the threshold
def get_manual_threshold(errors: np.ndarray, percentage: int) -> float:
    """
    Calculate the threshold based on the errors.

    Parameters:
    errors (List[float]): The errors.
    percentage (int): The percentile to use for the threshold.

    Returns:
    float: The calculated threshold.
    """
    return np.percentile(errors, percentage)

# Function to evaluate the autoencoder model
def evaluate_autoencoder(autoencoder: Model, validation_generator: ImageDataGenerator, test_generator: ImageDataGenerator, config, wandb = None) -> None:
    """
    Evaluate the autoencoder model.

    Parameters:
        autoencoder (Model): The autoencoder model.
        validation_generator (ImageDataGenerator): The validation data generator.
        test_generator (ImageDataGenerator): The test data generator.
        wandb: Weights and Biases object for logging.
        config: Configuration object containing loss function and other parameters.
    """
    # Step 1: Calculate threshold based on the validation set
    validation_errors, _ = get_errors_and_labels(
        autoencoder=autoencoder, 
        generator=validation_generator, 
        loss_function=config['loss']
    )

    threshold = get_manual_threshold(
        errors=validation_errors, 
        percentage=config['threshold_percentage']
    )

    if wandb: wandb.log({"selected_threshold_logic": "manual"})
    if wandb: wandb.log({"threshold": threshold})
    print(f"Manual Threshold used: {threshold:.4f}")

    # Step 2: Get errors and labels for the test set
    test_errors, true_labels = get_errors_and_labels(
        autoencoder=autoencoder, 
        generator=test_generator, 
        loss_function=config['loss']
    )

    # Step 3: Predict labels based on the threshold
    predicted_labels = np.where(test_errors > threshold, 1, 0)

    # Step 4: Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Step 5: Calculate F1 score
    f1 = f1_score(true_labels, predicted_labels, labels=ground_truth_labels)
    if wandb: wandb.log({"f1_score": f1})

    # Step 6: Split errors based on the true labels
    normal_errors = test_errors[true_labels == 0]
    anomalous_errors = test_errors[true_labels == 1]

    # Step 7: Plot error distribution for normal and anomalous samples
    plot_double_histogram_with_threshold(
        normal_errors,
        anomalous_errors,
        threshold,
        f"Reconstruction Error Distribution - Test Set - {config['comment']}",
        "Reconstruction Error",
        "Frequency",
        f"Threshold: {threshold:.4f}",
        wandb
    )

    # Step 8: Plot confusion matrix
    plot_confusion_matrix(conf_matrix, ground_truth_labels, f"Confusion Matrix - Test Set - {config['comment']}", wandb=wandb)

    # Step 9: Plot ROC curve and calculate AUC
    plot_roc_curve(true_labels, test_errors, f"ROC Curve - Test Set - {config['comment']}", wandb=wandb)

## Two next functions are used to evaluate the AE based on distributions and the sampled test set 
def evaluate_autoencoder_with_threshold_generator(autoencoder, test_generator, threshold_generator, validation_generator,config, wandb=None):
    """
    Evaluate the autoencoder using a threshold computed from the threshold generator.

    Args:
        autoencoder (Model): The autoencoder model.
        test_generator (ImageDataGenerator): The test data generator.
        threshold_generator (ImageDataGenerator): The threshold data generator.
        validation_generator (ImageDataGenerator): The validation data generator.
        config: Configuration object containing loss function and other parameters.
        wandb: Weights and Biases object for logging.
    """
    # Step 1: Calculate threshold from the threshold generator
    threshold = get_dist_based_threshold_between_spikes(
        autoencoder=autoencoder,
        threshold_generator=threshold_generator,
        loss_function=config['loss'],
        wandb=wandb,
        validation_generator=validation_generator,
        test_generator=test_generator,
        config=config
    )
    if threshold is None:
        return

    if wandb: wandb.log({"selected_threshold_logic": "automated"})
    if wandb: wandb.log({"threshold": threshold})
    print(f"Automated Threshold: {threshold:.4f}")

    # Step 2: Get errors and labels for the test set
    test_errors, true_labels = get_errors_and_labels(
        autoencoder=autoencoder,
        generator=test_generator,
        loss_function=config['loss']
    )

    # Step 3: Predict labels based on the threshold
    predicted_labels = np.where(test_errors > threshold, 1, 0)

    # Step 4: Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Step 5: Calculate F1 score
    f1 = f1_score(true_labels, predicted_labels, labels=ground_truth_labels)
    if wandb: wandb.log({"f1_score": f1})

    # Step 6: Split errors based on the true labels
    normal_errors = test_errors[true_labels == 0]
    anomalous_errors = test_errors[true_labels == 1]

    # Step 7: Plot error distribution for normal and anomalous samples
    plot_double_histogram_with_threshold(
        normal_errors,
        anomalous_errors,
        threshold,
        f"Reconstruction Error Distribution - Test Set - {config['comment']}",
        "Reconstruction Error",
        "Frequency",
        f"Threshold: {threshold:.4f}",
        wandb=wandb
    )

    # Step 8: Plot confusion matrix
    plot_confusion_matrix(
        confusion_matrix=conf_matrix, 
        labels=ground_truth_labels, 
        title=f"Confusion Matrix - Test Set - {config['comment']}",
        wandb=wandb)

    # Step 9: Plot ROC curve and calculate AUC
    plot_roc_curve(true_labels, test_errors, f"ROC Curve - Test Set - {config['comment']}", wandb=wandb)

# Function to evaluate the autoencoder based on the distribution of errors
def get_dist_based_threshold_between_spikes(
    autoencoder, 
    threshold_generator, 
    validation_generator, 
    test_generator, 
    wandb, 
    config, 
    loss_function, 
    num_steps=1000
):
    """
    Calculate the optimal threshold using the minimum between the spikes of normal and anomaly distributions.

    Args:
        autoencoder: Trained autoencoder model.
        threshold_generator (ImageDataGenerator): Generator for the threshold dataset.
        validation_generator (ImageDataGenerator): Validation dataset generator.
        test_generator (ImageDataGenerator): Test dataset generator.
        wandb: Weights & Biases logger instance.
        config: Configuration object containing model parameters.
        loss_function (str): Loss function for error calculation ('mse', 'mae').
        num_steps (int): Number of steps for evaluating KDE overlap.

    Assumption: 
        Distribution peak of anomalous samples is on the right of the distribution peak of good samples.
    """
    # Step 1: Calculate errors and labels from threshold generator
    errors, labels = get_errors_and_labels(autoencoder, threshold_generator, loss_function)

    # Step 2: Separate normal and anomaly errors
    normal_errors = errors[labels == 0]
    anomaly_errors = errors[labels == 1]

    # Step 3: KDE with Gaussian Kernel Density Estimation
    normal_kde = gaussian_kde(normal_errors)
    anomaly_kde = gaussian_kde(anomaly_errors)

    # Define the x-axis for KDE evaluation
    x = np.linspace(errors.min(), errors.max(), num_steps)

    # Evaluate densities across the x-axis
    normal_density = normal_kde(x)
    anomaly_density = anomaly_kde(x)

    # Find density peaks
    normal_peak_index = np.argmax(normal_density)
    anomaly_peak_index = np.argmax(anomaly_density)


    # Step 4: Check assumption and determine threshold
    if anomaly_peak_index <= normal_peak_index:
        print(f"Warning - Assumption violated: Anomaly peak is not to the right of normal peak.")
        print("Triggering `evaluate_autoencoder` as fallback evaluation.")

        plot_smooth_error_distribution(
        x=x,
        normal_density=normal_density,
        anomaly_density=anomaly_density,
        threshold=None,  # No threshold if assumption fails
        normal_peak_index=normal_peak_index,
        anomaly_peak_index=anomaly_peak_index,
        wandb=wandb
    )
        
        evaluate_autoencoder(
            autoencoder=autoencoder,
            validation_generator=validation_generator,
            test_generator=test_generator,
            wandb=wandb,
            config=config
        )
        return None
    else:
        # Define the region between the spikes
        x_between_spikes = x[normal_peak_index:anomaly_peak_index]
        kde_overlap_between_spikes = np.abs(normal_kde(x_between_spikes) - anomaly_kde(x_between_spikes))

        # Find the threshold in this region
        optimal_threshold_index = np.argmin(kde_overlap_between_spikes)
        threshold = x_between_spikes[optimal_threshold_index]

        # Plot smooth distribution with the threshold
        plot_smooth_error_distribution(
            x=x,
            normal_density=normal_density,
            anomaly_density=anomaly_density,
            threshold=threshold,
            normal_peak_index=normal_peak_index,
            anomaly_peak_index=anomaly_peak_index,
            wandb=wandb
        )

        print(f"Optimal Threshold Found: {threshold:.4f}")
        return threshold


## Unused: Function to plot reconstruction with original image and mask
def get_dist_based_threshold(autoencoder, threshold_generator, loss_function, num_steps=1000):
    """
    Calculate the optimal threshold using the minimum between the spikes of normal and anomaly distributions.

    Args:
        autoencoder: Trained autoencoder model.
        threshold_generator (ImageDataGenerator): Generator for the threshold dataset.
        loss_function (str): Loss function for error calculation ('mse', 'mae').
        num_steps (int): Number of steps for evaluating KDE overlap.
        bandwidth (float): Bandwidth for KDE.

    Assumption: 
        Distribution peak of anomalous samples is on the right of the distribution peak of good samples. 
    """
    errors, labels = [], []

    # Iterate through the threshold generator to process all images
    for batch_images, batch_labels in threshold_generator:
        reconstructions = autoencoder.predict(batch_images, verbose=0)
        
        batch_errors = calculate_error(batch_images, reconstructions, loss_function)
    
        # Append errors and labels
        errors.extend(batch_errors)
        labels.extend(batch_labels)

        # Stop when we've processed the entire generator
        if len(errors) >= threshold_generator.samples:
            break

    # Convert to numpy arrays
    errors = np.array(errors)
    labels = np.argmax(np.array(labels), axis=1)  # Convert one-hot to class indices

    # Separate normal and anomaly errors
    normal_errors = errors[labels == 0]
    anomaly_errors = errors[labels == 1]

    # KDE with bandwidth adjustment
    normal_kde = gaussian_kde(normal_errors)
    anomaly_kde = gaussian_kde(anomaly_errors)

    # Define the x-axis for KDE evaluation (entire range of errors)
    x = np.linspace(errors.min(), errors.max(), num_steps)

    # Find peaks (spikes) in the distributions
    normal_density = normal_kde(x)
    anomaly_density = anomaly_kde(x)

    normal_peak_index = np.argmax(normal_density)
    anomaly_peak_index = np.argmax(anomaly_density)

    # Ensure that the anomaly spike is to the right of the normal spike
    if anomaly_peak_index <= normal_peak_index:
        raise ValueError("Assumption violated: Anomaly peak is not to the right of normal peak.")

    # Define the region between the spikes
    x_between_spikes = x[normal_peak_index:anomaly_peak_index]
    kde_overlap_between_spikes = np.abs(normal_kde(x_between_spikes) - anomaly_kde(x_between_spikes))

    # Find the threshold in this region
    optimal_threshold_index = np.argmin(kde_overlap_between_spikes)
    threshold = x_between_spikes[optimal_threshold_index]

    return threshold


### Evaluate with KNN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, 
    f1_score, 
    roc_auc_score, 
    precision_recall_curve, 
    auc, 
    accuracy_score, 
    roc_curve
)
import numpy as np
from tensorflow.keras.models import Model


def extract_latent_vectors_and_labels(autoencoder, generator, layer_name='bottleneck'):
    """
    Extract latent vectors and labels from a generator using the autoencoder.
    """
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(layer_name).output)
    latent_vectors = []
    labels = []

    steps = len(generator)
    for i in range(steps):
        batch_images, batch_labels = next(generator)
        latent = encoder.predict(batch_images, verbose=0)
        
        latent_vectors.append(latent.reshape(latent.shape[0], -1))
        labels.append(np.argmax(batch_labels, axis=1))
    
    latent_vectors = np.concatenate(latent_vectors)
    labels = np.concatenate(labels)
    
    return latent_vectors, labels


def calculate_knn_distances(latent_vectors, n_neighbors=5):
    """
    Train kNN on latent vectors and return average kNN distances.
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(latent_vectors)
    distances, _ = knn.kneighbors(latent_vectors)
    avg_knn_distances = distances.mean(axis=1)
    return avg_knn_distances


def evaluate_autoencoder_with_KNN(autoencoder, validation_generator, test_generator, 
                                  layer_name='bottleneck', anomaly_percentile=95, k_neighbors=5, config=None, wandb=None):
    """
    Evaluate an Autoencoder for anomaly detection using kNN distances in the latent space.

    Parameters:
        autoencoder (Model): Trained autoencoder model.
        validation_generator (Iterator): Data generator for validation data (only normal samples).
        test_generator (Iterator): Data generator for test data (normal + anomaly).
        layer_name (str): Name of the bottleneck layer in the autoencoder.
        anomaly_percentile (float): Percentile threshold for anomaly detection.
        k_neighbors (int): Number of neighbors for kNN.
        config (dict): Configuration dictionary for labeling and comments.
        wandb: Weights & Biases logger instance (optional).

    Returns:
        dict: Dictionary containing evaluation metrics and plots.
    """
    ## 🟢 **Step 1: Extract Latent Vectors from Validation Set (Normal Only)**
    val_latent_vectors, _ = extract_latent_vectors_and_labels(autoencoder, validation_generator, layer_name)
    
    scaler = StandardScaler()
    val_latent_vectors_normalized = scaler.fit_transform(val_latent_vectors)
    
    # Calculate kNN distances on validation set
    val_knn_distances = calculate_knn_distances(val_latent_vectors_normalized, n_neighbors=k_neighbors)
    
    # Set threshold based on validation data
    threshold = np.percentile(val_knn_distances, anomaly_percentile)
    print(f"✅ Threshold set from validation set: {threshold:.4f}")
    if wandb: wandb.log({"kNN_threshold": threshold})
    
    ## 🟡 **Step 2: Extract Latent Vectors from Test Set (Normal + Anomaly)**
    test_latent_vectors, test_labels = extract_latent_vectors_and_labels(autoencoder, test_generator, layer_name)
    test_latent_vectors_normalized = scaler.transform(test_latent_vectors)
    
    # Calculate kNN distances on test set
    test_knn_distances = calculate_knn_distances(test_latent_vectors_normalized, n_neighbors=k_neighbors)
    
    # Binary predictions based on the threshold
    predictions = (test_knn_distances > threshold).astype(int)
    
    # Map ground-truth labels
    if 'good' in test_generator.class_indices:
        good_label_index = test_generator.class_indices['good']
        true_labels = (test_labels != good_label_index).astype(int)
    else:
        raise ValueError("Ensure that the test generator contains a 'good' class for normal samples.")
    
    ## 📊 **Step 3: Evaluate Performance**
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    auc_score = roc_auc_score(true_labels, test_knn_distances)
    precision, recall, _ = precision_recall_curve(true_labels, test_knn_distances)
    pr_auc = auc(recall, precision)
    cm = confusion_matrix(true_labels, predictions)
    
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")
    print(f"✅ AUC: {auc_score:.4f}")
    print(f"✅ PR-AUC: {pr_auc:.4f}")
    print("✅ Confusion Matrix:\n", cm)
    
    ## 📈 **Step 4: Visualization**
    # Plot Confusion Matrix
    plot_confusion_matrix(
        confusion_matrix=cm, 
        labels=['Normal', 'Anomaly'], 
        title=f"Confusion Matrix - Test Set - {config['comment']}",
        wandb=wandb
    )
    
    # Plot ROC Curve
    plot_roc_curve(
        true_labels, 
        test_knn_distances, 
        title=f"ROC Curve - Test Set - {config['comment']}"
    )
    
    # Plot Anomaly Score Distribution
    normal_distances = test_knn_distances[true_labels == 0]
    anomalous_distances = test_knn_distances[true_labels == 1]
    
    plot_double_histogram_with_threshold(
        normal_distances,
        anomalous_distances,
        threshold,
        f"Anomaly Score Distribution - Test Set - {config['comment']}",
        "Anomaly Score",
        "Frequency",
        f"Threshold: {threshold:.4f}"
    )





### Make predictions   
def predict_anomaly(
    image_path: str,
    evaluation_method: Literal['autoencoder', 'threshold_generator', 'KNN'],
    autoencoder: Model,
    validation_generator=None,
    test_generator=None,
    threshold_generator=None,
    config=None,
    wandb=None
) -> str:
    """
    Predict whether an image is an anomaly using a selected evaluation method.

    Parameters:
        image_path (str): Path to the input image.
        evaluation_method (str): Evaluation method ('autoencoder', 'threshold_generator', 'KNN').
        autoencoder (Model): Trained autoencoder model.
        validation_generator (ImageDataGenerator, optional): Validation data generator.
        test_generator (ImageDataGenerator, optional): Test data generator.
        threshold_generator (ImageDataGenerator, optional): Threshold data generator.
        config (dict, optional): Configuration dictionary.
        wandb (object, optional): WandB logger.

    Returns:
        str: Prediction result - "Anomaly" or "Normal".
    """
    # Step 1: Load and preprocess the image
    img = image.load_img(image_path, target_size=(config['image_size'], config['image_size']))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize to [0, 1]
    
    if evaluation_method == 'autoencoder':
        print("Using Autoencoder Evaluation Method...")
        reconstruction = autoencoder.predict(img_array, verbose=0)
        error = np.mean(np.square(img_array - reconstruction))
        
        # Calculate threshold from validation set
        validation_errors, _ = get_errors_and_labels(
            autoencoder=autoencoder,
            generator=validation_generator,
            loss_function=config['loss']
        )
        threshold = get_manual_threshold(validation_errors, config['threshold_percentage'])
        
        prediction = "Anomaly" if error > threshold else "Normal"
        print(f"Reconstruction Error: {error:.4f}, Threshold: {threshold:.4f}")
    
    elif evaluation_method == 'threshold_generator':
        print("Using Threshold Generator Evaluation Method...")
        threshold = get_dist_based_threshold_between_spikes(
            autoencoder=autoencoder,
            threshold_generator=threshold_generator,
            validation_generator=validation_generator,
            test_generator=test_generator,
            config=config,
            wandb=wandb,
            loss_function=config['loss']
        )
        
        reconstruction = autoencoder.predict(img_array, verbose=0)
        error = np.mean(np.square(img_array - reconstruction))
        
        prediction = "Anomaly" if error > threshold else "Normal"
        print(f"Reconstruction Error: {error:.4f}, Threshold: {threshold:.4f}")
    
    elif evaluation_method == 'KNN':
        print("Using KNN Evaluation Method...")
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(config['bottleneck_layer']).output)
        latent = encoder.predict(img_array, verbose=0)
        latent = latent.reshape(1, -1)
        
        # Fit KNN on the test dataset
        latent_vectors = []
        for i in range(len(test_generator)):
            batch_images, _ = next(test_generator)
            batch_latent = encoder.predict(batch_images, verbose=0)
            latent_vectors.append(batch_latent.reshape(batch_latent.shape[0], -1))
        
        latent_vectors = np.concatenate(latent_vectors)
        knn = NearestNeighbors(n_neighbors=config['n_neighbors'])
        knn.fit(latent_vectors)
        
        distances, _ = knn.kneighbors(latent)
        avg_knn_distance = np.mean(distances)
        
        reconstruction = autoencoder.predict(img_array, verbose=0)
        reconstruction_error = np.mean(np.square(img_array - reconstruction))
        
        # Normalize both metrics
        errors = np.concatenate([reconstruction_errors for _, reconstruction_errors in get_errors_and_labels(autoencoder, test_generator, config['loss'])])
        avg_knn_distances = np.concatenate([np.mean(knn.kneighbors(batch_latent.reshape(batch_latent.shape[0], -1))[0], axis=1) for batch_latent in latent_vectors])
        
        reconstruction_error = (reconstruction_error - np.min(errors)) / (np.max(errors) - np.min(errors))
        avg_knn_distance = (avg_knn_distance - np.min(avg_knn_distances)) / (np.max(avg_knn_distances) - np.min(avg_knn_distances))
        
        anomaly_score = reconstruction_error + avg_knn_distance
        threshold = np.percentile(anomaly_score, config['anomaly_percentile'])
        
        prediction = "Anomaly" if anomaly_score > threshold else "Normal"
        print(f"Anomaly Score: {anomaly_score:.4f}, Threshold: {threshold:.4f}")
    
    else:
        raise ValueError("Invalid evaluation method. Choose from 'autoencoder', 'threshold_generator', or 'KNN'.")
    
    print(f"Prediction: {prediction}")
    return prediction
