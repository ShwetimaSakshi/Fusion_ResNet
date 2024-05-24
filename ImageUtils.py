import numpy as np
from matplotlib import pyplot as plt
import cv2

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    ### YOUR CODE HERE
    # reshaping the image to unflatten it into [depth, height, width]
    # and convert it into [height, width, depth] form
    unflattend_image = record.reshape((3, 32, 32))
    image = np.transpose(unflattend_image, [1, 2, 0])
    ### END CODE HERE

    image = preprocess_image(image, training) # If any.

    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [3, 32, 32].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32]. The processed image.
    """
    ### YOUR CODE HERE

    if training:
        # adding padding of size 4 on all sides
        # cropping a portion of size 32,32 randomly from the image
        # flipping the image horizontally if the flipping probability>0.8 
        padding_size= ((4, 4), (4, 4), (0, 0))
        image = np.pad(image, padding_size, mode='constant')
        crop_start_x = np.random.randint(0, 9)
        crop_start_y = np.random.randint(0, 9)
        image = image[crop_start_y:crop_start_y+32, crop_start_x:crop_start_x+32, :]
        fliping_probability = np.random.rand()
        if(fliping_probability>0.8):
            image = np.fliplr(image)
            
        # random rotation
        angle = np.random.randint(-10, 10)
        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
        
        image = random_erasing(image)
    
    # Mean value for the Red channel: 0.49139967861519607
    # Mean value for the Green channel: 0.48215840839460783
    # Mean value for the Blue channel: 0.44653091444546567
    # Standard deviation for the Red channel: 0.2470322324632819
    # Standard deviation for the Green channel: 0.24348512800005573
    # Standard deviation for the Blue channel: 0.26158784172796434
    total_mean = (255 * np.array([0.49139, 0.48215, 0.44653])).reshape((3,1,1))
    total_std = (255 * np.array([0.24703, 0.2434, 0.2615])).reshape((3,1,1))
    image = (np.transpose(image, [2, 0, 1]) - total_mean) / total_std

    ### END CODE HERE

    return image

def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    
    # reshape to unflatten and convert into [height, width, depth] form
    unflattend_image = image.reshape((3, 32, 32))
    image = np.transpose(unflattend_image, [1, 2, 0])
    
    ### YOUR CODE HERE
    
    plt.imshow(image)
    plt.savefig(save_name)
    return image

# Other functions
### YOUR CODE HERE
def random_erasing(image):
    """Randomly select a rectange portion in the image and erase it 

    Args:
    image : An array of shape [3, 32, 32].

    Returns:
        Image with random erasing implemented over it.
    """
    # defining the parameters
    # probability to erase = 0.5
    # minimum proportion of erased part to the image = 0.02
    # maximum proportion of erased part to the image = 0.4
    # minimum aspect ration = 0.3
    # mean height, width and depth with which these are filled = [0.0, 0.0, 0.0]
    prob=0.5
    min_prop=0.02
    max_prop=0.4
    min_asp_ratio=0.3
    filler_value=[0.0, 0.0, 0.0]
    if np.random.rand() > prob:
        return image
    h, w, d = image.shape
    while True:
        target_area = np.random.uniform(min_prop, max_prop) * (h*w)
        aspect_ratio = np.random.uniform(min_asp_ratio, 1/min_asp_ratio)
        h_target = int(np.round(np.sqrt(target_area * aspect_ratio)))
        w_target = int(np.round(np.sqrt(target_area / aspect_ratio)))
        if w_target < w and h_target < h:
            x1 = np.random.randint(0, w - w_target)
            y1 = np.random.randint(0, h - h_target)
            break
    image[y1:y1 + h_target, x1:x1 + w_target] = np.random.normal(filler_value, (1.0, 1.0, 1.0), (h_target, w_target, 3))

    return image

import numpy as np
import random

def cutmix(image_batch, label_batch, batch_size, alpha=1.0):
    """
    Apply CutMix augmentation to a batch of images and labels.

    Args:
        image_batch: A batch of images of shape [batch_size, height, width, channels].
        label_batch: A batch of labels.
        batch_size: The size of the batch.
        alpha: The hyperparameter alpha for the beta distribution.

    Returns:
        mixed_images: A batch of mixed images.
        mixed_labels: A batch of mixed labels.
    """
    mixed_images = []
    mixed_labels = []

    for i in range(batch_size):
        # Select a random image and label from the batch
        image = image_batch[i]
        label = label_batch[i]

        # Select another random image and label from the batch
        j = random.randint(0, batch_size - 1)
        mixed_image = image_batch[j]
        mixed_label = label_batch[j]

        # Generate a random region to cut and mix
        height, width, _ = image.shape
        cut_ratio = np.sqrt(1.0 - np.random.beta(alpha, alpha))
        cut_height = int(height * cut_ratio)
        cut_width = int(width * cut_ratio)
        x1 = np.random.randint(0, height - cut_height + 1)
        y1 = np.random.randint(0, width - cut_width + 1)
        x2 = x1 + cut_height
        y2 = y1 + cut_width

        # Apply CutMix by replacing a region of the image with a region from another image
        mixed_image[x1:x2, y1:y2, :] = image[x1:x2, y1:y2, :]
        mixed_labels.append(label)

        mixed_images.append(mixed_image)

    mixed_images = np.array(mixed_images)
    mixed_labels = np.array(mixed_labels)

    return mixed_images, mixed_labels

### END CODE HERE

