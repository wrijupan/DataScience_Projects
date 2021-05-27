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


##########################################################
# Import Custom-made libraries
##########################################################
sys.path += ['src']
import utils as utils


def make_train_test_split(BASE, data_dir, test_size):
    """
    Split the data into train and test samples.

    The class labels are assigned following the convention-
    1: malaria
    0: no malaria

    The test data contains a balanced class.
    The remainning examples are used as train data.

    :param BASE: Base directory of data challenge.
    :param data_dir: Data directory
    :param test_size: Size of the test sample (0 to 1)
    :return: X_train, y_train, X_test, y_test
    """
    malaria_files = utils.get_malaria_files(BASE,
                                            data_dir)

    no_malaria_files = utils.get_no_malaria_files(BASE,
                                                  data_dir)

    if len(malaria_files) == 0:
        return [], [], [], []
    elif len(no_malaria_files) == 0:
        return [], [], [], []

    # Total number of examples (malaria and no malaria)
    total_number_of_files = len(malaria_files) + len(no_malaria_files)

    # Number of test samples to take from each of malaria and no malaria
    test_file_number = int(np.floor(0.5 * test_size
                                    * total_number_of_files))

    # Remainning number of train samples in both malaria and no malaria
    rem_train_malaria_files = len(malaria_files) - test_file_number
    rem_train_no_malaria_files = len(no_malaria_files) - test_file_number

    # Randomly sample test files
    random.seed(42)
    test_files = random.sample(malaria_files, test_file_number) \
        + random.sample(no_malaria_files, test_file_number)

    # Train files
    train_files = [f for f in malaria_files if f not in test_files] \
        + [f for f in no_malaria_files if f not in test_files]

    # Test labels
    test_labels = [1] * test_file_number \
        + [0] * test_file_number

    # Train labels
    train_labels = [1] * rem_train_malaria_files \
        + [0] * rem_train_no_malaria_files

    return train_files, train_labels, test_files, test_labels
