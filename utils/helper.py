import os
import keras
import random
import platform
import numpy as np
import tensorflow as tf

def get_root_dir() -> str:
    """
    Get the root directory of the repository.

    This function traverses up the directory tree until it finds the root directory
    of the repository, identified by the presence of a '.git' directory.

    Returns:
    str: The root directory of the repository.
    """
    current_dir = os.getcwd()
    
    # Traverse up the directory tree until you find the root directory of the repo
    while not os.path.exists(os.path.join(current_dir, '.git')):
        current_dir = os.path.dirname(current_dir)

    return current_dir

def set_seed(seed: int) -> None:
    """
    Set the random seed for numpy, random, TensorFlow, and Keras to ensure reproducibility.

    Parameters:
    seed (int): The seed value to set for random number generation.
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)

def setup_gpu() -> None:
    """
    Configures TensorFlow to use GPU if available, otherwise defaults to CPU.

    This function checks for available GPUs and sets memory growth to avoid
    TensorFlow from allocating all GPU memory at once. It also handles
    platform-specific configurations for macOS and other platforms.

    Raises:
        RuntimeError: If there is an error in setting memory growth.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            if platform.system() == 'Darwin':  # macOS
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print("Configured TensorFlow to use Metal on macOS.")
            else:  # Assume CUDA for other platforms
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print("Configured TensorFlow to use CUDA.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found, using CPU.")
