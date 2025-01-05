from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from typing import Tuple
from scipy.stats import gaussian_kde

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
def evaluate_autoencoder(autoencoder: Model, validation_generator: ImageDataGenerator, test_generator: ImageDataGenerator, wandb, config, ) -> None:
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
        loss_function=config.loss
    )

    threshold = get_manual_threshold(
        errors=validation_errors, 
        percentage=config.threshold_percentage
    )

    wandb.log({"threshold": threshold})
    print(f"Manual Threshold used: {threshold:.4f}")

    # Step 2: Get errors and labels for the test set
    test_errors, true_labels = get_errors_and_labels(
        autoencoder=autoencoder, 
        generator=test_generator, 
        loss_function=config.loss
    )

    # Step 3: Predict labels based on the threshold
    predicted_labels = np.where(test_errors > threshold, 1, 0)

    # Step 4: Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Step 5: Calculate F1 score
    f1 = f1_score(true_labels, predicted_labels, labels=ground_truth_labels)
    wandb.log({"f1_score": f1})

    # Step 6: Split errors based on the true labels
    normal_errors = test_errors[true_labels == 0]
    anomalous_errors = test_errors[true_labels == 1]

    # Step 7: Plot error distribution for normal and anomalous samples
    plot_double_histogram_with_threshold(
        normal_errors,
        anomalous_errors,
        threshold,
        f"Reconstruction Error Distribution - Test Set - {config.comment}",
        "Reconstruction Error",
        "Frequency",
        f"Threshold: {threshold:.4f}",
        wandb
    )

    # Step 8: Plot confusion matrix
    plot_confusion_matrix(conf_matrix, ground_truth_labels, f"Confusion Matrix - Test Set - {config.comment}", wandb=wandb)

    # Step 9: Plot ROC curve and calculate AUC
    plot_roc_curve(true_labels, test_errors, f"ROC Curve - Test Set - {config.comment}", wandb=wandb)

## Two next functions are used to evaluate the AE based on distributions and the sampled test set 
def evaluate_autoencoder_with_threshold_generator(autoencoder, test_generator, threshold_generator, validation_generator,config, wandb):
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
        loss_function=config.loss,
        wandb=wandb,
        validation_generator=validation_generator,
        test_generator=test_generator,
        config=config
    )
    if threshold is None:
        return

    wandb.log({"threshold": threshold})
    print(f"Optimal Threshold: {threshold:.4f}")

    # Step 2: Get errors and labels for the test set
    test_errors, true_labels = get_errors_and_labels(
        autoencoder=autoencoder,
        generator=test_generator,
        loss_function=config.loss
    )

    # Step 3: Predict labels based on the threshold
    predicted_labels = np.where(test_errors > threshold, 1, 0)

    # Step 4: Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Step 5: Calculate F1 score
    f1 = f1_score(true_labels, predicted_labels, labels=ground_truth_labels)
    wandb.log({"f1_score": f1})

    # Step 6: Split errors based on the true labels
    normal_errors = test_errors[true_labels == 0]
    anomalous_errors = test_errors[true_labels == 1]

    # Step 7: Plot error distribution for normal and anomalous samples
    plot_double_histogram_with_threshold(
        normal_errors,
        anomalous_errors,
        threshold,
        f"Reconstruction Error Distribution - Test Set - {config.comment}",
        "Reconstruction Error",
        "Frequency",
        f"Threshold: {threshold:.4f}",
        wandb=wandb
    )

    # Step 8: Plot confusion matrix
    plot_confusion_matrix(
        confusion_matrix=conf_matrix, 
        labels=ground_truth_labels, 
        title=f"Confusion Matrix - Test Set - {config.comment}",
        wandb=wandb)

    # Step 9: Plot ROC curve and calculate AUC
    plot_roc_curve(true_labels, test_errors, f"ROC Curve - Test Set - {config.comment}", wandb=wandb)

# Function to evaluate the autoencoder based on the distribution of errors
def get_dist_based_threshold_between_spikes(autoencoder, threshold_generator,validation_generator,test_generator, wandb, config, loss_function, num_steps=1000):
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
    # Iterate through the threshold generator to process all images
    errors, labels = get_errors_and_labels(autoencoder, threshold_generator, loss_function)

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
        print(f"Warning - Assumption violated: Anomaly peak is not to the right of normal peak.")
        print("Triggering `evaluate_autoencoder` as fallback evaluation.")
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

        plot_smooth_error_distribution(
            x = x,
            normal_density = normal_density,
            anomaly_density = anomaly_density,
            threshold=threshold,
            normal_peak_index=normal_peak_index,
            anomaly_peak_index=anomaly_peak_index,
            wandb=wandb
        )

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