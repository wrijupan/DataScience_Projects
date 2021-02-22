# MRI Brain Tumor detection
The aim of this project is to develop a Deep Learning Convolutional Neural Network to classify MRI images whether they show brain tumor or not. The original data set for this work has been collected from Kaggle.com.

Here the aim is to use Transfer Learning to develop a CNN for application to image classification task. Tensorflow-Keras provides a convenient access to some of the top performing models on the ImageNet image recognition tasks such as VGG.

The training time on the CPU is quite exaustive. This project is still under development and hyperparameter optimization for the CNN model will be refined by performing computation on a GPU (Google Colab provides 12 hours of free GPU access).

# Original data set
https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection

# Project overview
The steps involved in the project will be the following:

1. Setting up the environment
2. Loading the data and splitting it into train, test and validation.

Train data: Used to train the CNN model
Validation data: Used to check the model performance while training
Test data: An independent data set used to check the model performance

3. Data pre-processing
A. Finding the brain image contours and extreme points of the contour, to crop out only the brain part from the entire image
B. Data normalization to have a mean of zero and std. deviation of 1 for each individual color channel in each image.

4. Building the Deep CNN model architecture
5. Training and validating the model
6. Testing the model for production.
