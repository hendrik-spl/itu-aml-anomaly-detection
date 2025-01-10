import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data_with_test_split(category: str, batch_size: int, test_split: float = 0.5, rotation_range: int = 0):

    #check if the first path exists, otherwise use the second path
    data_dir = f'data/{category}'
    if not os.path.exists(data_dir):
        data_dir = f'../data/{category}'
    if not os.path.exists(data_dir):
        data_dir = f'../../data/{category}'
    train_dir = f'{data_dir}/train'
    test_dir = f'{data_dir}/test'

    #training generator with data augmentation
    datagen_train = ImageDataGenerator(
        rescale=1./255, 
        validation_split=0.2,
        rotation_range=rotation_range,
    )

    # validation generator without data augmentation
    datagen_val = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # test and threshold (validation of test set) generators
    datagen_test = ImageDataGenerator(
        rescale=1./255,
        validation_split=test_split  
    )

    train_generator = datagen_train.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='input',
        color_mode='rgb',
        subset='training'
    )

    validation_generator = datagen_val.flow_from_directory(
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
