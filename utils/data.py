import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data_with_test_split(category: str, batch_size: int, test_split: float = 0.5):
    """
    Generates data generators for training, validation, testing, and threshold datasets.

    Args:
        category (str): The category of the dataset to be used.
        batch_size (int): The size of the batches of data.
        test_split (float): Fraction of the test set to use for validation (threshold set).

    Returns:
        Tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator, DirectoryIterator]: 
        A tuple containing the training data generator, validation data generator, 
        test data generator, and threshold (validation of test set) data generator.
    """
    data_dir = f'../../data/{category}'
    train_dir = f'{data_dir}/train'
    test_dir = f'{data_dir}/test'

    # Training and validation generators
    datagen_train = ImageDataGenerator(
        rescale=1./255, 
        validation_split=0.2
    )

    # Test and threshold (validation of test set) generators
    datagen_test = ImageDataGenerator(
        rescale=1./255,
        validation_split=test_split  # Split test set into test and threshold sets
    )

    train_generator = datagen_train.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='input',
        color_mode='rgb',
        subset='training'
    )

    validation_generator = datagen_train.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='input',
        color_mode='rgb',
        subset='validation'
    )

    test_generator = datagen_test.flow_from_directory(
        test_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        subset='training'  # Subset for the actual test set
    )

    threshold_generator = datagen_test.flow_from_directory(
        test_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        subset='validation'  # Subset for the threshold set
    )

    return train_generator, validation_generator, test_generator, threshold_generator
