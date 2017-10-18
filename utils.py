

"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import math
import json
import random
import scipy.misc
import numpy as np
from time import gmtime, strftime
# -----------------------------
# new added functions for pix2pix

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
import pdb

def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.
    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]


def process_mnist(fname = "MNIST_data/mnist.mat"):

	d = sio.loadmat(fname)
	train_data = d['train_data']
	test_data = d['test_data']
	train_label = d['train_label']
	test_label = d['test_label']
	train_data = train_data * 2.0- 1.0
	test_data = test_data * 2.0 - 1.0
	sio.savemat( fname, {\
		"train_data" : train_data, \
		"test_data" : test_data, \
		"train_label" : train_label,\
		"test_label" : test_label})

def plot(samples, img_dim = 32):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(img_dim, img_dim), cmap='Greys_r')

    return fig

def download_preprocess(fname = "MNIST_data/mnist.mat"):
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	train_data = mnist.train.images *2.0 -1.0
	test_data = mnist.test.images *2.0 -1.0
	train_label = mnist.train.labels
	test_label = mnist.test.labels
	sio.savemat( fname, {\
		"train_data" : train_data, \
		"test_data" : test_data, \
		"train_label" : train_label,\
		"test_label" : test_label})


def load_data(image_path, flip=True, is_test=False):
    img_A, img_B = load_image(image_path)
    img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def load_image(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/2)
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w]

    return img_A, img_B

def preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    else:
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    return img_A, img_B

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    if len( images.shape) == 2:
        shapes = images.shape
        # images = images.reshape( (shapes[0], int( math.sqrt(shapes[1])), int( math.sqrt(shapes[1])) ,1 ) )
        images = images.reshape( (shapes[0], 28, 28 ,1 ) )
    
    h, w = images.shape[1], images.shape[2]
    if len(images.shape) == 4 and images.shape[-1 ] == 3:
    	channel = 3
        img = np.zeros((h * size[0], w * size[1], channel))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h, i*w:i*w+w, :] = image

        return img

    else:
    	channel = 1
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            # if images.shape != 3:
                # image = image.reshape( (28, 28))
            # pdb.set_trace()
            img[j*h:j*h+h, i*w:i*w+w] = image[:, :, 0]
        return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

