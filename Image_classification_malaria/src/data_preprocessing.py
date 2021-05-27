##########################################################
# Import Tensorflow libraries
##########################################################
import tensorflow as tf
from tensorflow.keras import models, layers


def make_tf_dataset(X_train, y_train):
    """
    Make Tensorflow constant from a list
    of image files (X) and image labels.

    :param X_train (list): Paths to the JPEG files
    :param y_train (list): Labels corresponding to JPEG
    :return: Tensorflow datasets using X_train and y_train
    """
    X_train_tfconst = tf.constant(X_train)
    y_train_tfconst = tf.constant(y_train)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_tfconst,
                                                        y_train_tfconst))

    return train_dataset


def encode(train_image, train_label, image_size=(40, 40)):
    """
    Read the JPEG files and image labels.
    Convert them in float32.
    Resize images to image_size.

    :param train_image: Sample  image
    :param train_label: Sample image label
    :param image_size: (height, width) for image resizing
    """
    # Read JPEG files and preprocess image
    image_string = tf.io.read_file(train_image)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size)
    # Preprocess image label
    label = tf.cast(train_label, tf.float32)
    return image, label


@tf.function
def data_augumentation(image, label):
    """
    Perform data augumentation by performing random
    image operations.

    :param image: Sample image to augument
    :param label: Image label
    """
    augumented_image = image
    augumented_image = tf.image.random_flip_left_right(augumented_image)
    augumented_image = tf.image.random_flip_up_down(augumented_image)
    augumented_image = tf.image.random_contrast(augumented_image, 0.5, 1.0)
    augumented_image = tf.image.random_brightness(augumented_image, 0.25)
    augumented_image = tf.clip_by_value(augumented_image, 0.0, 1.0)

    return augumented_image, label
