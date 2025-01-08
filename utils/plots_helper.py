import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns

def plot_double_histogram_with_threshold(normal_errors, anomaly_errors, threshold: float, title: str, xlabel: str, ylabel: str, threshold_label: str, wandb = None) -> None:
    plt.figure(figsize=(8, 6))
    plt.hist(normal_errors, bins=50, alpha=0.5, label='Normal')
    plt.hist(anomaly_errors, bins=50, alpha=0.5, label='Anomaly')
    plt.axvline(threshold, color='r', linestyle='--', label=threshold_label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if wandb: wandb.log({"error_distr_plot": wandb.Image(plt)})
    plt.show()

def plot_confusion_matrix(confusion_matrix: np.ndarray, labels, title: str, wandb = None) -> None:
    plt.figure(figsize=(8, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    if wandb: wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.show()

def plot_roc_curve(true_labels: np.ndarray, predicted_scores: np.ndarray, title: str, wandb = None) -> None:
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
    auc = roc_auc_score(true_labels, predicted_scores)
    if wandb: wandb.log({"auc": auc})
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='red')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    if wandb: wandb.log({"roc_curve": wandb.Image(plt)})
    plt.show()

def plot_smooth_error_distribution(x: np.ndarray, normal_density: np.ndarray, anomaly_density: np.ndarray, threshold: float, normal_peak_index: int, anomaly_peak_index: int, wandb = None) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(x, normal_density, label='Normal Errors', color='blue')
    plt.plot(x, anomaly_density, label='Anomaly Errors', color='orange')
    if threshold: plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.scatter(x[normal_peak_index], normal_density[normal_peak_index], color='blue', label='Normal Peak')
    plt.scatter(x[anomaly_peak_index], anomaly_density[anomaly_peak_index], color='orange', label='Anomaly Peak')
    plt.title('Error Distributions with Optimal Threshold Between Spikes')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.legend()
    if wandb: wandb.log({"distribution_with_thresholds": wandb.Image(plt)})
    plt.show()