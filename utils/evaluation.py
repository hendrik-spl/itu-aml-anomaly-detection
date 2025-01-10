from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

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

    all_errors = []
    all_labels = []
    
    generator.reset()

    for _ in range(len(generator)):
        batch_images, batch_labels = next(generator)
        reconstructions = autoencoder.predict(batch_images, verbose=0)
        batch_errors = calculate_error(batch_images, reconstructions, loss_function)
        

        all_errors.extend(batch_errors)
        all_labels.extend(batch_labels[:, relevant_label_index])
    
    return np.array(all_errors), np.array(all_labels)

# Function to calculate errors and labels
def get_errors_and_labels(autoencoder: Model, generator: ImageDataGenerator, loss_function: str) -> Tuple[np.ndarray, np.ndarray]:

    relevant_label_index = generator.class_indices["good"] 

    errors, labels_good = extract_errors_and_labels_from_generator(
        autoencoder=autoencoder,
        generator=generator,
        loss_function=loss_function,
        relevant_label_index=relevant_label_index
    )

    labels = 1 - labels_good

    return errors, labels

def get_manual_threshold(errors: np.ndarray, percentage: int) -> float:

    return np.percentile(errors, percentage)

def evaluate_autoencoder(autoencoder: Model, validation_generator: ImageDataGenerator, test_generator: ImageDataGenerator, config, wandb = None) -> None:
    
    #Calculate threshold based on the validation set
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

    #get errors and labels for the test set
    test_errors, true_labels = get_errors_and_labels(
        autoencoder=autoencoder, 
        generator=test_generator, 
        loss_function=config['loss']
    )

    # Predict labels based on the threshold
    predicted_labels = np.where(test_errors > threshold, 1, 0)

    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    f1 = f1_score(true_labels, predicted_labels, labels=ground_truth_labels)
    if wandb: wandb.log({"f1_score": f1})

    normal_errors = test_errors[true_labels == 0]
    anomalous_errors = test_errors[true_labels == 1]

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

    plot_confusion_matrix(conf_matrix, ground_truth_labels, f"Confusion Matrix - Test Set - {config['comment']}", wandb=wandb)

    plot_roc_curve(true_labels, test_errors, f"ROC Curve - Test Set - {config['comment']}", wandb=wandb)


def evaluate_autoencoder_with_threshold_generator(autoencoder, test_generator, threshold_generator, validation_generator,config, wandb=None):

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

    test_errors, true_labels = get_errors_and_labels(
        autoencoder=autoencoder,
        generator=test_generator,
        loss_function=config['loss']
    )

    predicted_labels = np.where(test_errors > threshold, 1, 0)

    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    f1 = f1_score(true_labels, predicted_labels, labels=ground_truth_labels)
    if wandb: wandb.log({"f1_score": f1})

    normal_errors = test_errors[true_labels == 0]
    anomalous_errors = test_errors[true_labels == 1]

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
    num_steps=1000,
    task=None
):
    # calculate errors and labels from threshold generator
    errors, labels = get_errors_and_labels(autoencoder, threshold_generator, loss_function)

    #separate normal and anomaly errors
    normal_errors = errors[labels == 0]
    anomaly_errors = errors[labels == 1]

    #Gaussian Kernel Density Estimation
    normal_kde = gaussian_kde(normal_errors)
    anomaly_kde = gaussian_kde(anomaly_errors)

    # Define the x-axis for KDE evaluation
    x = np.linspace(errors.min(), errors.max(), num_steps)

    #evaluate densities across the x-axis
    normal_density = normal_kde(x)
    anomaly_density = anomaly_kde(x)

    #find density peaks
    normal_peak_index = np.argmax(normal_density)
    anomaly_peak_index = np.argmax(anomaly_density)


    # check assumption and determine threshold
    if anomaly_peak_index <= normal_peak_index:
        print(f"Warning - Assumption violated: Anomaly peak is not to the right of normal peak.")
        print("Triggering `evaluate_autoencoder` as fallback evaluation.")

        if task != 'predict':
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
        # define the region between the spikes
        x_between_spikes = x[normal_peak_index:anomaly_peak_index]
        kde_overlap_between_spikes = np.abs(normal_kde(x_between_spikes) - anomaly_kde(x_between_spikes))

        # find the threshold in this region
        optimal_threshold_index = np.argmin(kde_overlap_between_spikes)
        threshold = x_between_spikes[optimal_threshold_index]

        #plot smooth distribution with the threshold
        if task != 'predict':
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

    errors, labels = [], []

    for batch_images, batch_labels in threshold_generator:
        reconstructions = autoencoder.predict(batch_images, verbose=0)
        
        batch_errors = calculate_error(batch_images, reconstructions, loss_function)
    
        errors.extend(batch_errors)
        labels.extend(batch_labels)

        if len(errors) >= threshold_generator.samples:
            break

    errors = np.array(errors)
    labels = np.argmax(np.array(labels), axis=1)

    # Separate normal and anomaly errors
    normal_errors = errors[labels == 0]
    anomaly_errors = errors[labels == 1]

    # KDE with bandwidth adjustment
    normal_kde = gaussian_kde(normal_errors)
    anomaly_kde = gaussian_kde(anomaly_errors)

    #define the x-axis for KDE evaluation (entire range of errors)
    x = np.linspace(errors.min(), errors.max(), num_steps)

    # Find peak in the distributions
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

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(latent_vectors)
    distances, _ = knn.kneighbors(latent_vectors)
    avg_knn_distances = distances.mean(axis=1)
    return avg_knn_distances


def evaluate_autoencoder_with_KNN(autoencoder, validation_generator, test_generator, 
                                  layer_name='bottleneck', anomaly_percentile=95, k_neighbors=5, config=None, wandb=None):

    ## extract Latent Vectors from Validation Set (Normal Only)
    val_latent_vectors, _ = extract_latent_vectors_and_labels(autoencoder, validation_generator, layer_name)
    
    scaler = StandardScaler()
    val_latent_vectors_normalized = scaler.fit_transform(val_latent_vectors)
    
    # Calculate knn distances on validation set
    val_knn_distances = calculate_knn_distances(val_latent_vectors_normalized, n_neighbors=k_neighbors)
    
    # Set threshold based on validation data
    threshold = np.percentile(val_knn_distances, anomaly_percentile)
    print(f"Threshold set from validation set: {threshold:.4f}")
    if wandb: wandb.log({"kNN_threshold": threshold})
    
    ##extract Latent Vectors from Test Set
    test_latent_vectors, test_labels = extract_latent_vectors_and_labels(autoencoder, test_generator, layer_name)
    test_latent_vectors_normalized = scaler.transform(test_latent_vectors)
    
    # Calculate knn distances on test set
    test_knn_distances = calculate_knn_distances(test_latent_vectors_normalized, n_neighbors=k_neighbors)
    
    # Binary predictions based on the threshold
    predictions = (test_knn_distances > threshold).astype(int)
    
    # Map ground-truth labels
    if 'good' in test_generator.class_indices:
        good_label_index = test_generator.class_indices['good']
        true_labels = (test_labels != good_label_index).astype(int)
    else:
        raise ValueError("Ensure that the test generator contains a 'good' class for normal samples.")
    
    ##evaluate Performance
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    auc_score = roc_auc_score(true_labels, test_knn_distances)
    precision, recall, _ = precision_recall_curve(true_labels, test_knn_distances)
    pr_auc = auc(recall, precision)
    cm = confusion_matrix(true_labels, predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc_score:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print("Confusion Matrix:\n", cm)
    
    plot_confusion_matrix(
        confusion_matrix=cm, 
        labels=['Normal', 'Anomaly'], 
        title=f"Confusion Matrix - Test Set - {config['comment']}",
        wandb=wandb
    )
    
    plot_roc_curve(
        true_labels, 
        test_knn_distances, 
        title=f"ROC Curve - Test Set - {config['comment']}"
    )
    
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

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0 
    
    if evaluation_method == 'autoencoder':
        print("\nUsing Autoencoder Evaluation Method...")
        reconstruction = autoencoder.predict(img_array, verbose=0)
        error = np.mean(np.square(img_array - reconstruction))
        
        validation_errors, _ = get_errors_and_labels(
            autoencoder=autoencoder,
            generator=validation_generator,
            loss_function=config['loss']
        )
        threshold = get_manual_threshold(validation_errors, config['threshold_percentage'])
        
        prediction = "Anomaly" if error > threshold else "Normal"
        print(f"Reconstruction Error: {error:.4f}, Threshold: {threshold:.4f}")
    
    elif evaluation_method == 'threshold_generator':
        print("\nUsing Threshold Generator Evaluation Method...")
        threshold = get_dist_based_threshold_between_spikes(
            autoencoder=autoencoder,
            threshold_generator=threshold_generator,
            validation_generator=validation_generator,
            test_generator=test_generator,
            config=config,
            wandb=wandb,
            loss_function=config['loss'],
            task='predict'
        )
        
        reconstruction = autoencoder.predict(img_array, verbose=0)
        error = np.mean(np.square(img_array - reconstruction))
        
        prediction = "Anomaly" if error > threshold else "Normal"
        print(f"Reconstruction Error: {error:.4f}, Threshold: {threshold:.4f}")
    
    elif evaluation_method == 'KNN':
        print("\nUsing KNN Evaluation Method...")
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('bottleneck').output)
        latent = encoder.predict(img_array, verbose=0)
        latent = latent.reshape(1, -1)
        
        # Extract latent vectors from the validation generator
        val_latent_vectors, _ = extract_latent_vectors_and_labels(autoencoder, validation_generator, 'bottleneck')
        scaler = StandardScaler()
        val_latent_vectors_normalized = scaler.fit_transform(val_latent_vectors)
        
        knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        knn.fit(val_latent_vectors_normalized)
        latent_normalized = scaler.transform(latent)
        distances, _ = knn.kneighbors(latent_normalized)
        avg_knn_distance = np.mean(distances)
        
        threshold = np.percentile(calculate_knn_distances(val_latent_vectors_normalized, n_neighbors=5), 80)
        
        prediction = "Anomaly" if avg_knn_distance > threshold else "Normal"
        print(f"kNN Distance: {avg_knn_distance:.4f}, Threshold: {threshold:.4f}")
    
    else:
        raise ValueError("Invalid evaluation method. Choose from 'autoencoder', 'threshold_generator', or 'KNN'.")
    
    print(f"Prediction: {prediction}")
    return prediction
