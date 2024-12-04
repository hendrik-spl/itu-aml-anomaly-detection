from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def load_data(category: str, batch_size: int):
    """
    Generates data generators for training, validation, and testing datasets.

    Args:
        category (str): The category of the dataset to be used.
        batch_size (int): The size of the batches of data.

    Returns:
        Tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator]: 
        A tuple containing the training data generator, validation data generator, and test data generator.
    """
    data_dir = f'../../data/{category}'
    train_dir = f'{data_dir}/train'
    test_dir = f'{data_dir}/test'

    datagen_train = ImageDataGenerator(
        rescale=1./255, 
        validation_split=0.2
        )

    datagen_test = ImageDataGenerator(
        rescale=1./255
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
        shuffle=False
    )

    return train_generator, validation_generator, test_generator

def threshold_data_loader(data_generator, batch_size: int, threshold_split: float = 0.5):
    """
    Dynamically samples from the test directory to create a threshold dataset.

    Args:
        category (str): The category of the dataset.
        batch_size (int): The batch size for the data loader.
        threshold_split (float): Fraction of the dataset to use for threshold calculation.

    Returns:
        DirectoryIterator: A data loader for the threshold dataset.
    """    

    # Create a generator for the test directory
    test_generator = data_generator

    # Calculate the total number of images to sample for threshold setting
    total_images = test_generator.samples
    threshold_sample_size = int(total_images * threshold_split)

    # Generate a subset of the test set for threshold calculation
    sampled_images, sampled_labels = [], []
    for i in range(threshold_sample_size // batch_size + 1):
        batch_images, batch_labels = next(test_generator)
        sampled_images.append(batch_images)
        sampled_labels.append(batch_labels)

    # Combine sampled batches
    sampled_images = np.concatenate(sampled_images, axis=0)[:threshold_sample_size]
    sampled_labels = np.concatenate(sampled_labels, axis=0)[:threshold_sample_size]

    return sampled_images, sampled_labels

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
