"""This is a module for functions related to testing the perturbed data for MNIST"""

from __future__ import division

import argparse
import os
import getpass
if getpass.getuser() == 'aria':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import tensorflow as tf
import numpy as np
from setup_mnist import MNISTModel, OddEvenMNIST
from Dataset2 import odd_even_labels
from sklearn.metrics import accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default="models/",
        help="Path to save trained model.")
    parser.add_argument(
        '--pert_data',
        type=str,
        default='./MNIST_data/perturbed.npz',
        help='Path to LFW perturbed data.')
    parser.add_argument(
        '--orig_data',
        type=str,
        default='MNIST_data/B.npz',
        help="Path to original data."
    )
    parser.add_argument(
        '--image_size', type=int, default=28, help='Size of input images.')
    parser.add_argument(
        '--num_channels',
        type=int,
        default=1,
        help='Number of channels in input images.')
    parser.add_argument(
        '--train_new',
        dest='train_new',
        action='store_true',
        help='Train a new classifier.')
    parser.set_defaults(train_new=False)
    args = parser.parse_args()

    loaded = np.load(args.pert_data)
    pert_data = np.concatenate((loaded['train_data'], loaded['test_data']))
    pert_data = pert_data.reshape(pert_data.shape[0], args.image_size, args.image_size, -1)
    pert_evil_label = np.concatenate((loaded['train_label'], loaded['test_label'])).\
        argmax(axis=1)
    pert_good_label = odd_even_labels(pert_evil_label).\
        argmax(axis=1)

    loaded = np.load(args.orig_data)
    orig_data = np.concatenate((loaded['train_data'], loaded['test_data']))
    orig_data = orig_data.reshape(orig_data.shape[0], args.image_size, args.image_size, -1)
    orig_evil_label = np.concatenate((loaded['train_label'], loaded['test_label'])).\
        argmax(axis=1)
    orig_good_label = odd_even_labels(orig_evil_label).\
        argmax(axis=1)
    print 'Original data shape:', orig_data.shape

    good_used = OddEvenMNIST(args.model_path + 'A_odd_even')
    good_left = OddEvenMNIST(args.model_path + 'C_odd_even')
    evil_used = MNISTModel(args.model_path + 'A_digits')
    evil_left = MNISTModel(args.model_path + 'C_digits')

    evil_pair = (orig_evil_label, pert_evil_label)
    good_pair = (orig_good_label, pert_good_label)

    for model, label_pair, name in zip(
            [evil_used, good_used, evil_left, good_left],
            [evil_pair, good_pair, evil_pair, good_pair],
            ['Used Evil', 'Used Good', 'Left-out Evil', 'Left-out Good']):

        org_true, pert_true = label_pair
        print name + ':'
        org_pred = np.argmax(model.model.predict(orig_data), axis=1)
        print org_pred.shape
        print org_true.shape
        org_acc = accuracy_score(org_true, org_pred)
        print '\tOriginal Accuracy: %.4f' % org_acc
        dst_pred = np.argmax(model.model.predict(pert_data), axis=1)
        dst_acc = accuracy_score(pert_true, dst_pred)
        print '\tPerturbed Accuracy: %.4f' % dst_acc

    if args.train_new:
        # Train a new classifier with the new training data, test with original test data.
        raise NotImplementedError(
            'Training new classifier is not yet implemented.')


if __name__ == '__main__':
    main()
