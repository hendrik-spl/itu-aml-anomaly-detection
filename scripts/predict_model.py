import os
import sys
import wandb
import argparse
import tensorflow as tf
sys.path.insert(0, os.getcwd())

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.helper import set_seed
from utils.data import load_data_with_test_split
from utils.latent_space import plot_latent_space, plot_combined_latent_space
from utils.evaluation import evaluate_autoencoder, evaluate_autoencoder_with_threshold_generator, evaluate_autoencoder_with_KNN, predict_anomaly

# Here we parse the arguments to get the run_id of the model we want to evaluate
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--run_id", type=str, required=True, help="Name of the wandb run of the model to evaluate")
    return parser.parse_args()

args = parse_args()
run_id = args.run_id

def get_wandb_data(model_name):
    try:
        api = wandb.Api()
        run = api.run(f"ablation-study/{model_name}")
        name = run.name
        config = run.config
        return name, config
    except Exception as e:
        raise FileNotFoundError(f"Model wandb logs with name {model_name} not found. Error: {e}")

def get_model(model_name):
    model_path = f"../models/models/checkpoints/{model_name}.keras"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file with name {model_name} not found at path: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    return model

def main(run_id):
    set_seed(42)

    run_name, config = get_wandb_data(run_id)
    model = get_model(run_name)

    train_generator, validation_generator, test_generator, threshold_generator = load_data_with_test_split(
    category=config['data_class'],
    batch_size=32,
    test_split=0.4,
    rotation_range=config['rotation_range'],
    )

    if model is None or config is None:
        raise FileNotFoundError(f"Check failed: Model with name {run_name} not found")

    image_path = '../data/screw/test/scratch_head/000.png'

    predict_anomaly(
    image_path=image_path,
    evaluation_method='autoencoder',  # Options: 'autoencoder', 'threshold_generator', 'KNN'
    autoencoder=model,
    validation_generator=validation_generator,
    test_generator=test_generator,
    threshold_generator=threshold_generator,
    config=config
    )

    predict_anomaly(
    image_path=image_path,
    evaluation_method='threshold_generator',  # Options: 'autoencoder', 'threshold_generator', 'KNN'
    autoencoder=model,
    validation_generator=validation_generator,
    test_generator=test_generator,
    threshold_generator=threshold_generator,
    config=config
    )

    predict_anomaly(
    image_path=image_path,
    evaluation_method='KNN',  # Options: 'autoencoder', 'threshold_generator', 'KNN'
    autoencoder=model,
    validation_generator=validation_generator,
    test_generator=test_generator,
    threshold_generator=threshold_generator,
    config=config
    )

if __name__ == "__main__":
    main(run_id)
