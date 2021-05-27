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


def get_malaria_files(BASE, data_dir):
    """
    Get the list of malaria JPEG files.

    :param BASE (str): Base directory of data challenge
    :param data_dir (str): Data directory
    :return: List of malaria files
    """
    malaria_dir = os.path.join(BASE, data_dir, 'malaria')

    if not os.path.exists(malaria_dir):
        print("The directory {} does not exist...".format(malaria_dir))
        return []
    return sorted(glob.glob(os.path.join(malaria_dir, '*.jpg')))


def get_no_malaria_files(BASE, data_dir):
    """
    Get the list of no malaria JPEG files.

    :param BASE (str): Base directory of data challenge
    :param data_dir (str): Data directory
    :return: List of no malaria files
    """
    malaria_dir = os.path.join(BASE, data_dir, 'no malaria')

    if not os.path.exists(malaria_dir):
        print("The directory {} does not exist...".format(malaria_dir))
        return []
    return sorted(glob.glob(os.path.join(malaria_dir, '*.jpg')))


def plot_sample_images(dataset, num_samples):
    """
    Plot some sample images

    :param dataset: Dataset
    :param num_samples: No.of samples to plot
    """
    index = 1
    num_samples = num_samples
    plt.figure(figsize=(24, 4))

    for image, label in dataset.take(num_samples):
        plt.subplot(1, num_samples, index)
        plt.imshow(image)
        plt.title(f"Label: {label}\nShape: {image.shape}")
        plt.axis('off')
        index += 1
