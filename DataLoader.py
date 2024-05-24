import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    # the train data is loadded for 5 sets
    for i in range(1, 6):
        train_path = os.path.join(data_dir, f"data_batch_{i}")
        with open(train_path, 'rb') as file:
            dict = pickle.load(file, encoding='bytes')
            x_train.extend(dict[b'data'])
            y_train.extend(dict[b'labels'])
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # the public test data is loaded
    test_path = os.path.join(data_dir, "test_batch")
    with open(test_path, 'rb') as file:
        dict = pickle.load(file, encoding='bytes')
        x_test.extend(dict[b'data'])
        y_test.extend(dict[b'labels'])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    # Split RGB channels
    x_train_r = x_train[:, :1024]  # Red channel
    x_train_g = x_train[:, 1024:2048]  # Green channel
    x_train_b = x_train[:, 2048:]  # Blue channel

    # Calculate mean for each channel
    #mean_r = np.mean(x_train_r) / 255.0  # Divide by 255 to scale to [0, 1] range
    #mean_g = np.mean(x_train_g) / 255.0
    #mean_b = np.mean(x_train_b) / 255.0

    # Calculate standard deviation for each channel
    #std_r = np.std(x_train_r) / 255.0  # Divide by 255 to scale to [0, 1] range
    #std_g = np.std(x_train_g) / 255.0
    #std_b = np.std(x_train_b) / 255.0

    # Print mean and standard deviation values for each channel
    #print("Mean value for the Red channel:", mean_r)
    #print("Mean value for the Green channel:", mean_g)
    #print("Mean value for the Blue channel:", mean_b)
    #print("Standard deviation for the Red channel:", std_r)
    #print("Standard deviation for the Green channel:", std_g)
    #print("Standard deviation for the Blue channel:", std_b)
    
    ### END CODE HERE

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 3072].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    # the private test data is loaded
    x_test = np.load(data_dir)
    print('shape of private test data:', x_test)
    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    # the point where we split the data is calculated
    # training and validatiation dataset are separated
    l = int(len(x_train) * train_ratio)
    x_train_new = x_train[:l]
    y_train_new = y_train[:l]
    x_valid = x_train[l:]
    y_valid = y_train[l:]

    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid
