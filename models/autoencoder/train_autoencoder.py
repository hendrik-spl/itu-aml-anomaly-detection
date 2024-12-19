import os
import sys
sys.path.insert(0, os.getcwd())

import tensorflow as tf
from keras.callbacks import EarlyStopping

import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from utils.helper import get_root_dir, set_seed, setup_gpu
from utils.data import load_data_with_test_split
from utils.plots import plot_reconstructions, plot_history
from utils.latent_space import plot_latent_space
from utils.evaluation import evaluate_autoencoder, evaluate_autoencoder_with_threshold_generator
from utils.models import vanilla_autoencoder

config = {
        "comment" : "test py file",
        "epochs" : 1,
        "loss" : 'ssim', # available options: 'mse', 'mae', 'ssim', 'ssim_l1', 'dssim'
        "optimizer" : 'adam',
        "dropout_value" : 0.0,
        "rotation_range" : 90,
        "batch_size" : 16,
        "latent_dim" : 512,
        "data_class" : "screw",
        }

def main(config: dict):
    set_seed(1234)

    wandb.init(project="autoencoder", config=config)
    wandb.define_metric('val_loss', summary='min')
    config = wandb.config

    # Load data
    train_generator, validation_generator, test_generator, threshold_generator = load_data_with_test_split(
        category=config.data_class,
        batch_size=32,
        test_split=0.4,
        rotation_range=config.rotation_range
        )

    # Build model
    autoencoder = vanilla_autoencoder(
        input_shape=(256, 256, 3), 
        optimizer=config.optimizer,
        latent_dim=config.latent_dim, 
        loss=config.loss
    )

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', patience=20),
        WandbMetricsLogger(),
        WandbModelCheckpoint(filepath=f"models/checkpoints/{config.comment}.keras", verbose=1, save_best_only=True)
    ]

    # Train model
    history = autoencoder.fit(
        train_generator,
        epochs=config.epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Plot results
    plot_history(comment=config.comment, history=history, wandb=wandb)
    plot_reconstructions(autoencoder, test_generator, n_images=5, title='Test', wandb=wandb)

    # Evaluate model
    evaluate_autoencoder_with_threshold_generator(
        autoencoder=autoencoder,
        test_generator=test_generator,
        threshold_generator=threshold_generator,
        config=config,
        wandb=wandb
    )

    # Plot latent space
    plot_latent_space(autoencoder, test_generator, wandb, layer_name='bottleneck')

if __name__ == "__main__":    
    main(config=config)