# train_models.py -- train the neural network models for attacking
##
# Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
# This program is licenced under the BSD 2-Clause licence,
# contained in the LICENCE file in this directory.


import os
import opts
import numpy as np
import sys
# import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# from setup_mnist import MNIST
# from setup_cifar import CIFAR

mnist_flag = 1

def odd_even_labels(labels, one_hot=True):
    """
    Turns digit classes to odd[1]/even[0].
    Returns hot-one encoded labels.
    """
    if one_hot is True:
        labels = np.argmax(labels, axis=1)

    num_samples = labels.shape[0]
    new_labels = np.zeros((num_samples, 2))

    for mod_2 in range(2):
        # new_labels[even, 0] = 1
        # new_labels[odd, 1] = 1
        new_labels[labels % 2 == mod_2, mod_2] = 1

    return new_labels

def train(file_name, params, num_epochs=50,
          batch_size=128, train_temp=1, init=None, odd_even=False):
    """
    Standard neural network training procedure.
    """
    opt = opts.parse_opt()
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_data = mnist.train.images * 2.0 - 1.0
    train_label = mnist.train.labels


    test_data = mnist.test.images * 2.0 - 1.0
    test_label = mnist.test.labels

    if odd_even is True:
        train_label = odd_even_labels(train_label, one_hot=True)
        test_label = odd_even_labels(test_label, one_hot=True)

    x_dim = train_data.shape[1]
    y_dim = train_label.shape[1]
    img_dim = 28
    input_c_dim = 1
    output_c_dim = 1
    input_dim = x_dim
    label_dim = y_dim

    # Resizing input images.
    source = tf.placeholder(tf.float32, [None, input_dim], name="source_image")
    images = tf.reshape(source, [-1, img_dim, img_dim, input_c_dim])

    # Run the session to get the resized images.
    with tf.Session() as sess:
        new_train_data = sess.run(images, feed_dict={source: train_data})
        new_test_data = sess.run(images, {source: test_data})
        test_data = new_test_data
    print new_train_data.shape

    # Creating the Keras model for MNIST.
    model = Sequential()
    model.add(Conv2D(params[0], (3, 3),
                     input_shape=[img_dim, img_dim, input_c_dim]))
    model.add(Activation('relu'))
    model.add(Conv2D(params[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params[2], (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(params[3], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[4]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params[5]))
    model.add(Activation('relu'))
    # If odd/even is classified, y_dim = 2.
    # If all digits are classified, y_dim = 10.
    model.add(Dense(y_dim))
    # model.add(Activation('softmax'))

    # if init != None:
    #     model.load_weights(init)

    def fn(correct, predicted):
        """
        The loss function.
        """
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=correct, logits=predicted / train_temp)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(new_train_data, train_label,
              batch_size=batch_size,
              validation_data=(test_data, test_label),
              nb_epoch=num_epochs,
              shuffle=True)

    if file_name is not None:
        model.save(file_name)

    return model


if not os.path.isdir('models'):
    os.makedirs('models')

def main():
    if len(sys.argv) == 2:
        model_path = sys.argv[1]
    else:
        model_path = 'models/mnist'
    print 'Saving the model in %s' % model_path
    train(model_path, [32, 32, 64, 64, 200, 200], num_epochs=10)
    # train(CIFAR(), "models/cifar", [64, 64, 128, 128, 256, 256], num_epochs=50)
    # train( "models/cifar_model1", [64, 64, 128, 128, 256, 256], num_epochs=50)

if __name__ == '__main__':
    main()
