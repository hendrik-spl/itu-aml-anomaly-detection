from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
        subset='validation',
        shuffle=False
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
