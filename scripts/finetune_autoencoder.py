import os
import sys
sys.path.insert(0, os.getcwd())

import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint

import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.helper import set_seed
from utils.data import load_data_with_test_split
from utils.plots import plot_reconstructions, plot_history
from utils.latent_space import plot_latent_space
from utils.evaluation import evaluate_autoencoder_with_threshold_generator
from utils.models import get_model
from utils.loss import return_loss

wandb_project = "ablation-study"
wandb_tags = [
    "autoencoder",
    "reproducible",
    # "test" # remove this tage when running the actual training
] 

config = {
    "comment" : "mobilenet finetuning lr 1e-4",
    "model_name" : "mobilenet_autoencoder", # available options: "autoencoder", "mobilenet_autoencoder", "vanilla_autoencoder", "deep_autoencoder"
    "threshold_percentage": 80,
    # Taken as given
    "data_class" : "screw", # available options: "screw", "metal_nut" and more
    "epochs" : 1,
    "latent_dim" : 512,
    "optimizer" : 'adam',
    "downsampling": 'maxpooling',
    "bottleneck": 'dense',
    "batch_size" : 16,
    "rotation_range" : 90,
    # Parameters
    "decoder_type" : 'upsampling', # available options: 'upsampling', 'transposed'
    "num_blocks" : 8, # number of blocks in the encoder/decoder
    "batch_norm" : True, # available options: True, False
    "dropout_value" : 0.4, # setting this value to 0 will basically remove dropout layers
    "loss" : 'ssim', # available options: 'mae', 'mse', 'ssim'
}

def main(config):
    set_seed(42)

    wandb.init(project=wandb_project, tags=wandb_tags, config=config)
    wandb.define_metric('val_loss', summary='min')
    config = wandb.config

    checkpoint_path = f"../models/checkpoints/{wandb.run.name}.keras"
    
    # Load data
    train_generator, validation_generator, test_generator, threshold_generator = load_data_with_test_split(
        category=config.data_class,
        batch_size=32,
        test_split=0.4,
        rotation_range=config.rotation_range
        )

    # Load the best model (based on validation loss)
    pretrained_model_path = f"models/checkpoints/comic-gorge-110.keras"
    autoencoder = tf.keras.models.load_model(pretrained_model_path)

    # Log Model Size
    wandb.log({"model_size": autoencoder.count_params()})

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', patience=40),
        ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", mode="min", verbose=1, save_best_only=True),
        WandbMetricsLogger(),
    ]

    # Set adam optimizer to lower learning rate
    finetune_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # Finetune the model
    autoencoder.compile(optimizer=finetune_optimizer, loss=return_loss(config['loss']))
    history = autoencoder.fit(
        train_generator,
        epochs=config.epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Load the best model (based on validation loss)
    autoencoder = tf.keras.models.load_model(checkpoint_path)

    # Plot results
    plot_history(comment=config.comment, history=history, wandb=wandb)
    plot_reconstructions(autoencoder, test_generator, n_images=5, title='Test', wandb=wandb)

    # Evaluate model
    evaluate_autoencoder_with_threshold_generator(
        autoencoder=autoencoder,
        test_generator=test_generator,
        threshold_generator=threshold_generator,
        validation_generator=validation_generator,
        config=config,
        wandb=wandb
    )

    # Plot latent space
    plot_latent_space(autoencoder, test_generator, wandb, layer_name='bottleneck')

    # wandb.finish() # only needed for jupyter notebooks
    
if __name__ == "__main__":
    main(config=config)