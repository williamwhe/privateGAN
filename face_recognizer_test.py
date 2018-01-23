"""Testing module for face_recognizer.py"""
from __future__ import division

import time
import argparse

import numpy as np
import tensorflow as tf
from keras.optimizers import SGD

from lfw import get_30_people_chunk
from face_recognizer import FaceRecognizer

TEST_SGD = SGD(decay=1e-6, momentum=0.9, nesterov=True)
def TEST_SOFTMAX(correct, predicted):
    """
    The softmax loss function. For more than 2 one-hot encoded classes.
    """
    return tf.nn.softmax_cross_entropy_with_logits(
        labels=correct, logits=predicted)

def test_30_recognizer(args):
    print 'Identity Recognizer:'
    input_shape = (args.image_size, args.image_size, args.num_channels)
    X, y = get_30_people_chunk(args.image_path, 0)

    identity_recognizer = FaceRecognizer(args.model_path,
                                         y.shape[0],
                                         input_shape,
                                         args.num_channels)

    identity_recognizer.model.compile(loss=TEST_SOFTMAX,
                                      optimizer=TEST_SGD,
                                      metrics=['accuracy'])

    y_pred = identity_recognizer.model.predict(X)
    y_pred_ct = np.argmax(y_pred, axis=1)
    y_true_ct = np.argmax(y, axis=1)
    acc = np.sum(y_pred_ct == y_true_ct) / y_true_ct.shape[0]

    print '\tAccuracy on training data: %.4f' % acc

def test_gender_recognizer(args):
    input_shape = (args.image_size, args.image_size, args.num_channels)
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="models/lfw",
                        help="Path to save trained model. e.g.: 'models/lfw'")
    parser.add_argument('--image_path', type=str, default='./lfw_data/',
                        help='Path to LFW data.')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Size of input images.')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='Number of channels in input images.')
    parser.set_defaults(gender=False)
    args = parser.parse_args()

    test_30_recognizer(args)
    test_gender_recognizer(args)

if __name__ == '__main__':
    main()
