"""This is a module for functions related to testing the perturbed data"""

from __future__ import division

import argparse

import tensorflow as tf
import numpy as np
from face_recognizer import FaceRecognizer
from sklearn.metrics import accuracy_score

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
    parser.add_argument('--train_new', dest='train_new', action='store_true',
                        help='Train a new classifier.')
    parser.set_defaults(train_new=False)
    args = parser.parse_args()
    input_shape = (args.image_size, args.image_size, args.num_channels)

    data = np.load(args.image_path + 'perturbed.npz')
    id_gender = data['id_gender']

    org_train_data = data['org_train_data']
    train_data = data['train_data']
    train_id = np.argmax(data['train_label'], axis=1)
    train_gender = np.argmax(id_gender[train_id, :], axis=1)
    test_data = data['test_data']
    org_test_data = data['org_test_data']
    test_id = np.argmax(data['test_label'], axis=1)
    test_gender = np.argmax(id_gender[test_id, :], axis=1)
    num_good_labels = 2
    num_evil_labels = data['train_label'].shape[1]

    print train_data.shape
    print org_train_data.shape
    print test_data.shape
    print org_test_data.shape

    good_used = FaceRecognizer('%s_%d_gender_0' % (args.model_path, args.image_size),
                               num_good_labels,
                               input_shape)

    good_left = FaceRecognizer('%s_%d_gender_1' % (args.model_path, args.image_size),
                               num_good_labels,
                               input_shape)

    evil_used = FaceRecognizer('%s_%d_id_0' % (args.model_path, args.image_size),
                               num_evil_labels,
                               input_shape)

    evil_left = FaceRecognizer('%s_%d_id_1' % (args.model_path, args.image_size),
                               num_evil_labels,
                               input_shape)

    for model, label, name in zip([evil_used, good_used, evil_left, good_left],
                                  [test_id, test_gender, test_id, test_gender],
                                  ['Used Evil', 'Used Good', 'Left-out Evil', 'Left-out Good']):
        print name + ':'
        org_pred = np.argmax(model.predict(org_test_data), axis=1)
        org_acc = accuracy_score(label, org_pred)
        print '\tOriginal Accuracy: %.4f' % org_acc
        dst_pred = np.argmax(model.predict(test_data), axis=1)
        dst_acc = accuracy_score(label, dst_pred)
        print '\tPerturbed Accuracy: %.4f' % dst_acc

    if args.train_new:
        # Train a new classifier with the new training data, test with original test data.
        raise NotImplementedError('Training new classifier is not yet implemented.')

if __name__ == '__main__':
    main()
