# Detecting the symptoms of Parkinsons disease from the data collected using inertial measurement units: a combination of triaxial Accelerometer, Gyroscope and Magnetometer

This project consists of two parts:
1. Detecting the symptoms of dyskinesia (a particular type of symptom of the Parkinsons disease where there are strong uncontrolled movements of the body segments) using signal processing

2. Detecting the symptoms of dyskinesia using Machine Learning

# Data
The data set is collected using a wrist watch attached to the hand of a patient who is suffering from Parkinsons disease (strong or mild symptoms). The wrist watch contains inside it the combination of accelerometer, gyroscope of magnetometer. These devices monitor the movements in various parts of the body of the patient. By analyzing this movement data, I have shown that it is possible to identify periods when the patient is undergoing strong, mild or no symptoms of dyskinesia.

There are in total 5 data files. Each contains movement data from patients undergoing Parkinsons symptoms. The length of time for which data have been collected for each file is approximately 1 minute.

The main challenges in using Machine Learning in this particular data set is that it is a quite small data set and also contains imbalanced classes (dyskinesia or no dyskinesia). Hence to train a proper machine learning model much more data set representing different classes of patients is required. Here I have shown that signal processing can be used to identify periods of symptoms of dyskinesia. Then I have shown how to train a mock supervised Machine Learning model that tries to solve the same problem,i.e. identify periods of dyskinesia symptoms from the movement data of patients. Here as the labelled  data set for the supervised learning, I have used the labels obtained from the signal processing part.
