##########################################################
# Import Python libraries
##########################################################
import glob
import sys
import os
import tqdm
import time
import random
from collections import Counter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

##########################################################
# Import Tensorflow libraries
##########################################################
import tensorflow as tf
from tensorflow.keras import models, layers


@tf.function
def custom_step(tf_model, opt, X, y):
    """
    For each step, calclate the loss,
    calcuate the gradient of the loss using GradientTape
    and update the trainable model weights

    :param tf_model: Tensorflow model
    :param opt: Optimizer
    :param X: Train image batch
    :param y: Train image label
    :return: Loss for each step
    """
    with tf.GradientTape() as tape:
        y_pred = tf_model(X)
        y_pred = tf.reshape(y_pred, y.get_shape())

        loss = tf.keras.losses.binary_crossentropy(y, y_pred)

    gradient = tape.gradient(loss,
                             tf_model.trainable_variables
                             )
    opt.apply_gradients(zip(gradient,
                            tf_model.trainable_variables
                            ))

    return loss


def train_model(train_data, model, opt, Epochs, num_of_batches):
    """
    Train the model for n no.of epochs.

    If epoch number < 70% of total no. of epochs:
        Train the whole network

    If epoch number > 70% of total no. of epochs:
        Train only the Dense layers and freeze rest.

    :param Epochs (int): No. of epochs
    :param train_data: Tensorflow dataset
    :param model: CNN model
    :param opt: Optimizer
    """
    for epochs in range(Epochs):
        print("[INFO] Executing epoch {}/{}...".format(epochs + 1, Epochs))
        sys.stdout.flush()
        epochStart = time.time()

        epochLoss = 0.0

        if (epochs + 1) < (0.7 * Epochs):
            # While Number of epochs is less than 70% of total
            # Train the whole model
            for x_train, y_train in train_data:
                loss = custom_step(model,
                                   opt,
                                   x_train,
                                   y_train)
                epochLoss += loss

            epochEnd = time.time()
            elapsedTime = (epochEnd - epochStart) / 60.0
            print("Elapsed time={}".format(elapsedTime))

        else:
            # While Number of epochs is greater than 70% of total
            # Train only the Dense layers
            # First save the Dense layers and pop them out of the model
            output_layers = model.layers[-3:]
            model.pop()
            model.pop()
            model.pop()

            # Make the remainning layers (one's without the last 3)
            # non-trainable
            for layer in model.layers:
                layer.trainable = False

            # Add back the last 3 layers
            for output_layer in output_layers:
                model.add(output_layer)

            for x_train, y_train in train_data:
                loss = custom_step(model,
                                   opt,
                                   x_train,
                                   y_train)
                epochLoss += loss

            epochEnd = time.time()
            elapsedTime = (epochEnd - epochStart) / 60.0
            print("Elapsed time={}".format(elapsedTime))

        epochLoss /= num_of_batches
        print("Loss={}".format(epochLoss))
