##########################################################
# Import Tensorflow libraries
##########################################################
import tensorflow as tf
from tensorflow.keras import models, layers


def build_model(img_height, img_width, img_depth):
    """
    Build a Convolutional Neural Network model using
    Tensorflow.

    :param img_height (int): height of each img
    :param img_width (int): width of each img
    :param img_depth (int): No. of channels
    :return: A CNN model using Tensorflow
    """
    input_shape = (img_height, img_width, img_depth)

    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(filters=32,
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same',
                            input_shape=input_shape
                            ))

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(layers.Conv2D(filters=64,
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same'
                            ))

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(layers.Conv2D(filters=128,
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same'
                            ))

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten
    model.add(layers.Flatten())

    # Dense layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.4))

    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    return model
