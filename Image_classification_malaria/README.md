# Identifying the presence of malaria parasites in human cell images

## Project Description

Microscopy is the gold standard for the diagnosis of numerous communicable diseases worldwide, such as malaria. However, manually discriminating subtle differences on a stained slide under the light microscope is a labor-intensive, time-consuming and error-prone process.

Since the introduction of machine learning, there has been an increasing interest in applying them to improve different diagnostic technologies. Currently, machine learning is reaching a point it can classify images quickly and efficiently, even with lower image resolution.

In this project images of human cells are classified using Machine Learning, or more appropriately Convolutional Neural Networks (CNN), with the main aim of identifying the presence of malaria parasites in the cell images.

## Tasks

The main purpose of this project is not to build a high-level image classifier using the Keras library, but to leverage the power of Tensorflow 2.x to build a CNN model from scratch and make a customised training loop along with explicitly coded back-propagation. The following main tasks are accomplished

1. Performing Data Augumentation

2. Using the tf.function decorator

3. Using tf.GradientTape to perform back propagation

4. A customised training loop which trains the whole network for the first few epochs and then only trains the heads for the last epochs.


## Requirements
Python 3.7 or later and the following packages: 


`numpy`
`matplotlib`
`tensorflow`
`PIL`


## Instructions for use

Run the notebook 

`classify_malaria_cells.ipynb` 

that is found in the ‘notebooks’ directory. The notebook uses the following custom-made modules that are located in the ‘src’ directory.

1. cnn_model.py: Build a CNN model using Tensorflow

2. data_preprocessing.py: Converts the input image data into Tensorflow datasets and perform data argumentation.

3. model_training.py: Build a custom training loop with back-propagation using GradientTape and training the whole network for first few epochs and only the heads in the last epochs.

4. train_test_split.py: Divide the data into training (used for training the CNN model) and testing (used for evaluating the CNN model) sets

5. utils.py: Some helper functions for reading the data and plotting.