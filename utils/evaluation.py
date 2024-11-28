from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from scipy.stats import gaussian_kde

# https://medium.com/@majpaw1996/anomaly-detection-in-computer-vision-with-ssim-ae-2d5256ffc06b
def dssim_loss(y_true, y_pred):
    return 1/2 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))/2

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def ssim_l1_loss(y_true, y_pred, alpha=0.5):
    """
        y_true: Ground truth images.
        y_pred: Predicted images.
        alpha: Weighting factor for SSIM and L1 loss. 
               alpha = 0.5 means equal weight for both losses.
    """    
    # Compute SSIM loss
    ssim = tf.image.ssim(y_true, y_pred, 1.0)
    ssim_loss = 1 - tf.reduce_mean(ssim)
    
    # Compute L1 loss
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Combine SSIM and L1 losses
    combined_loss = alpha * ssim_loss + (1 - alpha) * l1_loss
    
    return combined_loss


def calculate_error(images: np.ndarray, reconstructions: np.ndarray, loss_function: str) -> List[float]:
    """
    Calculate error between original images and their reconstructions.

    Parameters:
    images (np.ndarray): The original images.
    reconstructions (np.ndarray): The reconstructed images.
    loss_function (str): The loss function to use ('mae', 'mse', 'dssim_loss', 'ssim_loss', 'ssim_l1_loss').

    Returns:
    List[float]: A list of errors for each image in the batch.
    """
    if loss_function == 'mae':
        return np.mean(np.abs(reconstructions - images), axis=(1, 2, 3)).tolist()
    elif loss_function == 'mse':
        return np.mean(np.square(reconstructions - images), axis=(1, 2, 3)).tolist()
    elif loss_function == 'dssim':
        dssim_values = 1 / 2 - tf.image.ssim(images, reconstructions, max_val=1.0) / 2
        return dssim_values.numpy().tolist()
    elif loss_function == 'ssim':
        ssim_values = 1 - tf.image.ssim(images, reconstructions, max_val=1.0)
        return ssim_values.numpy().tolist()
    elif loss_function == 'ssim_l1':
        # Compute SSIM for each image
        ssim = tf.image.ssim(images, reconstructions, 1.0)
        ssim_loss = 1 - ssim  # Batch-wise SSIM loss
        
        # Compute L1 loss for each image
        l1_loss = tf.reduce_mean(tf.abs(images - reconstructions), axis=(1, 2, 3))
        
        # Combine SSIM and L1 losses
        combined_loss = 0.5 * ssim_loss + 0.5 * l1_loss
        return combined_loss.numpy().tolist()
    else:
        raise ValueError(f"Unknown loss function: {loss_function}. Please define a function to calculate the error.")


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
    errors = []
    labels = []
    for i in range(len(generator)):
        batch_images, batch_labels = next(generator)
        reconstructions = autoencoder.predict(batch_images, verbose=0)
        batch_errors = calculate_error(batch_images, reconstructions, loss_function)
        errors.extend(batch_errors)
        labels.extend(batch_labels[:, relevant_label_index]) # this gets the relevant labels with 1 if it's a match for our folder (e.g. 'good') and 0 otherwise
    labels = [1 - label for label in labels] # invert the labels so that 1 is anomaly and 0 is normal
    return np.array(errors), np.array(labels)

def get_threshold(errors: np.ndarray, percentage: int) -> float:
    """
    Calculate the threshold based on the errors.

    Parameters:
    errors (List[float]): The errors.
    percentage (int): The percentile to use for the threshold.

    Returns:
    float: The calculated threshold.
    """
    return np.percentile(errors, percentage)

def plot_single_histogram_with_threshold(errors: List[float], threshold: float, title: str, xlabel: str, ylabel: str, threshold_label: str) -> None:
    """
    Plot a single histogram with a threshold line.

    Parameters:
    errors (List[float]): The errors to plot.
    threshold (float): The threshold value.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    threshold_label (str): The label for the threshold line.
    """
    plt.hist(errors, bins=50, alpha=0.5)
    plt.axvline(threshold, color='r', linestyle='--', label=threshold_label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_double_histogram_with_threshold(normal_errors: List[float], anomaly_errors: List[float], threshold: float, title: str, xlabel: str, ylabel: str, threshold_label: str) -> None:
    """
    Plot two histograms (normal and anomaly errors) with a threshold line.

    Parameters:
    normal_errors (List[float]): The normal errors to plot.
    anomaly_errors (List[float]): The anomaly errors to plot.
    threshold (float): The threshold value.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    threshold_label (str): The label for the threshold line.
    """
    plt.hist(normal_errors, bins=50, alpha=0.5, label='Normal')
    plt.hist(anomaly_errors, bins=50, alpha=0.5, label='Anomaly')
    plt.axvline(threshold, color='r', linestyle='--', label=threshold_label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_confusion_matrix(confusion_matrix: np.ndarray, labels: List[str], title: str) -> None:
    """
    Plot a confusion matrix.

    Parameters:
    confusion_matrix (np.ndarray): The confusion matrix to plot.
    labels (List[str]): The labels for the confusion matrix.
    title (str): The title of the plot.
    """
    plt.figure(figsize=(8, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def evaluate_autoencoder(autoencoder: Model, validation_generator: ImageDataGenerator, test_generator: ImageDataGenerator, config, threshold_type = 'simple') -> None:
    """
    Evaluate the autoencoder model.

    Parameters:
    autoencoder (Model): The autoencoder model.
    validation_generator (ImageDataGenerator): The validation data generator.
    test_generator (ImageDataGenerator): The test data generator.
    """
    ground_truth_labels = [
        "Normal", # 0
        "Anomaly" # 1
        ]

    # Step 1: Calculate reconstruction error on the validation set
    validation_errors, _ = get_errors_and_labels(
        autoencoder=autoencoder, 
        generator=validation_generator, 
        loss_function=config.loss
    )

    # Step 2: Calculate threshold based on the reconstruction error
    threshold = get_threshold(
        errors=validation_errors, 
        percentage=config.threshold_percentage
    )

    # Step 3: Plot error distribution with threshold
    plot_single_histogram_with_threshold(
        errors=validation_errors,
        threshold=threshold,
        title=f"Reconstruction Error Distribution - Validation Set - {config.comment}",
        xlabel="Reconstruction Error",
        ylabel="Frequency",
        threshold_label=f"Threshold at {config.threshold_percentage}th percentile: {threshold:.4f}",
    )

    # Step 4: Calculate test errors and labels
    test_errors, true_labels = get_errors_and_labels(
        autoencoder=autoencoder, 
        generator=test_generator, 
        loss_function=config.loss
    )

    # Step 6: Predict labels based on the threshold
    predicted_labels = np.where(test_errors > threshold, 1, 0)

    # Step 7: Calculate metrics
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Step 8: Split errors based on the true labels
    normal_errors = test_errors[true_labels == 0]
    anomalous_errors = test_errors[true_labels == 1]

    # Step 9: Plot error distribution for normal and anomalous samples
    plot_double_histogram_with_threshold(
        normal_errors,
        anomalous_errors,
        threshold,
        f"Reconstruction Error Distribution - Test Set - {config.comment}",
        "Reconstruction Error",
        "Frequency",
        f"Threshold at {config.threshold_percentage}th percentile: {threshold:.4f}",
    )
    print(classification_report(true_labels, predicted_labels, target_names=['Normal', 'Anomaly']))

    # Step 10: Plot confusion matrix
    plot_confusion_matrix(conf_matrix, ground_truth_labels, f"Confusion Matrix - Test Set - {config.comment}")


def evaluate_autoencoder_with_distribution_threshold(autoencoder: Model, 
                                                     test_generator: ImageDataGenerator, 
                                                     threshold_images: np.ndarray, 
                                                     threshold_labels: np.ndarray, 
                                                     config) -> None:
    """
    Evaluate the autoencoder model using a distribution-based threshold.

    Parameters:
        autoencoder (Model): The autoencoder model.
        test_generator (ImageDataGenerator): The test data generator.
        threshold_images (np.ndarray): Images used for calculating the threshold.
        threshold_labels (np.ndarray): Labels for the threshold dataset (0 for normal, 1 for anomaly).
        config: Configuration object containing loss function and other parameters.
    """
    ground_truth_labels = [
        "Normal",  # 0
        "Anomaly"  # 1
    ]

    # Step 1: Calculate the optimal threshold using error distributions
    threshold = get_dist_based_threshold(
        autoencoder=autoencoder,
        threshold_images=threshold_images,
        threshold_labels=np.argmax(threshold_labels, axis=1),  # Convert one-hot to class indices
        loss_function=config.loss
    )

    print(f"Optimal Threshold: {threshold}")

    # Step 2: Evaluate on the test set
    test_errors, true_labels = get_errors_and_labels(
        autoencoder=autoencoder, 
        generator=test_generator, 
        loss_function=config.loss
    )

    # Step 3: Predict labels based on the threshold
    predicted_labels = np.where(test_errors > threshold, 1, 0)

    # Step 4: Calculate metrics
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print(classification_report(true_labels, predicted_labels, target_names=ground_truth_labels))

    # Step 5: Split errors based on the true labels
    normal_errors = test_errors[true_labels == 0]
    anomalous_errors = test_errors[true_labels == 1]

    # Step 6: Plot error distributions for normal and anomalous samples
    plot_double_histogram_with_threshold(
        normal_errors,
        anomalous_errors,
        threshold,
        f"Reconstruction Error Distribution - Test Set - {config.comment}",
        "Reconstruction Error",
        "Frequency",
        f"Optimal Threshold: {threshold:.4f}"
    )

    # Step 7: Plot confusion matrix
    plot_confusion_matrix(conf_matrix, ground_truth_labels, f"Confusion Matrix - Test Set - {config.comment}")


def get_dist_based_threshold(autoencoder, threshold_images, threshold_labels, loss_function='mse', num_steps=1000):
    """
    Calculate the optimal threshold for separating normal and anomalous images.

    Args:
        autoencoder: Trained autoencoder model.
        threshold_images (np.ndarray): Images used for threshold calculation.
        threshold_labels (np.ndarray): Labels for the threshold images (0 for normal, 1 for anomaly).
        loss_function (str): Loss function for error calculation ('mse', 'mae').
        num_steps (int): Number of steps for evaluating KDE overlap.

    Returns:
        float: The calculated threshold value.
    """
    # Step 1: Reconstruct images
    reconstructed_images = autoencoder.predict(threshold_images)

    # Step 2: Calculate reconstruction errors
    if loss_function == 'mse':
        errors = np.mean((threshold_images - reconstructed_images) ** 2, axis=(1, 2, 3))
    elif loss_function == 'mae':
        errors = np.mean(np.abs(threshold_images - reconstructed_images), axis=(1, 2, 3))
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")

    # Step 3: Separate errors by label
    normal_errors = errors[threshold_labels == 0]
    anomaly_errors = errors[threshold_labels == 1]

    # Step 4: Estimate error distributions using Kernel Density Estimation
    normal_kde = gaussian_kde(normal_errors)
    anomaly_kde = gaussian_kde(anomaly_errors)

    # Step 5: Find the threshold that minimizes the overlap between distributions
    min_error = min(errors)
    max_error = max(errors)
    x = np.linspace(min_error, max_error, num_steps)
    kde_overlap = np.abs(normal_kde(x) - anomaly_kde(x))
    optimal_threshold_index = np.argmin(kde_overlap)
    threshold = x[optimal_threshold_index]

    # Step 6: Plot the distributions and threshold
    plt.figure(figsize=(8, 6))
    plt.plot(x, normal_kde(x), label='Normal Errors', color='blue')
    plt.plot(x, anomaly_kde(x), label='Anomaly Errors', color='orange')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Error Distributions with Optimal Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    return threshold

def get_dist_based_threshold_from_generator(autoencoder, threshold_generator, loss_function='mse', num_steps=1000):
    """
    Calculate the optimal threshold using a generator for the threshold dataset.

    Args:
        autoencoder: Trained autoencoder model.
        threshold_generator (ImageDataGenerator): Generator for the threshold dataset.
        loss_function (str): Loss function for error calculation ('mse', 'mae').
        num_steps (int): Number of steps for evaluating KDE overlap.

    Returns:
        float: The calculated threshold value.
    """
    import numpy as np
    from scipy.stats import gaussian_kde
    import matplotlib.pyplot as plt

    errors, labels = [], []

    # Iterate through the threshold generator to process all images
    for batch_images, batch_labels in threshold_generator:
        reconstructions = autoencoder.predict(batch_images, verbose=0)
        
        # Calculate reconstruction errors
        if loss_function == 'mse':
            batch_errors = np.mean((batch_images - reconstructions) ** 2, axis=(1, 2, 3))
        elif loss_function == 'mae':
            batch_errors = np.mean(np.abs(batch_images - reconstructions), axis=(1, 2, 3))
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
        
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

    # Estimate distributions using KDE
    normal_kde = gaussian_kde(normal_errors)
    anomaly_kde = gaussian_kde(anomaly_errors)

    # Find the threshold that minimizes the overlap between distributions
    min_error, max_error = min(errors), max(errors)
    x = np.linspace(min_error, max_error, num_steps)
    kde_overlap = np.abs(normal_kde(x) - anomaly_kde(x))
    optimal_threshold_index = np.argmin(kde_overlap)
    threshold = x[optimal_threshold_index]

    # Plot error distributions and threshold
    plt.figure(figsize=(8, 6))
    plt.plot(x, normal_kde(x), label='Normal Errors', color='blue')
    plt.plot(x, anomaly_kde(x), label='Anomaly Errors', color='orange')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Error Distributions with Optimal Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    return threshold

def evaluate_autoencoder_with_threshold_generator(autoencoder, test_generator, threshold_generator, config):
    """
    Evaluate the autoencoder using a threshold computed from the threshold generator.

    Args:
        autoencoder (Model): The autoencoder model.
        test_generator (ImageDataGenerator): The test data generator.
        threshold_generator (ImageDataGenerator): The threshold data generator.
        config: Configuration object containing loss function and other parameters.
    """
    # Calculate threshold from the threshold generator
    threshold = get_dist_based_threshold_from_generator(
        autoencoder=autoencoder,
        threshold_generator=threshold_generator,
        loss_function=config.loss
    )

    print(f"Optimal Threshold: {threshold:.4f}")

    # Evaluate on the test set
    test_errors, true_labels = get_errors_and_labels(
        autoencoder=autoencoder,
        generator=test_generator,
        loss_function=config.loss
    )

    # Predict labels based on the threshold
    predicted_labels = np.where(test_errors > threshold, 1, 0)

    # Compute metrics
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print(classification_report(true_labels, predicted_labels, target_names=['Normal', 'Anomaly']))

    # Split errors based on the true labels
    normal_errors = test_errors[true_labels == 0]
    anomalous_errors = test_errors[true_labels == 1]

    # Plot error distributions
    plot_double_histogram_with_threshold(
        normal_errors,
        anomalous_errors,
        threshold,
        f"Reconstruction Error Distribution - Test Set - {config.comment}",
        "Reconstruction Error",
        "Frequency",
        f"Threshold: {threshold:.4f}"
    )

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, ['Normal', 'Anomaly'], f"Confusion Matrix - Test Set - {config.comment}")
