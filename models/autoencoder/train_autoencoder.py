import os
import sys
sys.path.insert(0, os.getcwd())

import tensorflow as tf
from keras.callbacks import EarlyStopping

import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from utils.helper import set_seed
from utils.data import load_data_with_test_split
from utils.plots import plot_reconstructions, plot_history
from utils.latent_space import plot_latent_space
from utils.evaluation import evaluate_autoencoder_with_threshold_generator
from utils.models import get_model

wandb_project = "ablation-study"
wandb_tags = [
    "autoencoder", 
    "test" # remove this tage when running the actual training
]

config = {
        "comment" : "test dropout",
        "model_name" : "vanilla_autoencoder", # available options: "vanilla_autoencoder", "deep_autoencoder", ...
        # Taken as given
        "data_class" : "screw", # available options: "screw", "metal_nut" and more
        "epochs" : 200,
        "latent_dim" : 512,
        "optimizer" : 'adam',
        "batch_size" : 16,
        "rotation_range" : 90,
        # Hyperparameters
        "batch_norm" : True,
        "dropout_value" : 0.2, # setting this value to 0 will basically remove dropout layers
        "loss" : 'mae', # available options: 'mse', 'mae', 'ssim'
        }

def main(config):
    set_seed(42)

    wandb.init(project=wandb_project, tags=wandb_tags, config=config)
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
    autoencoder = get_model(config)

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