"""This is a module for functions related to testing the perturbed data"""

from __future__ import division

import argparse

import tensorflow as tf
import numpy as np
from face_recognizer import FaceRecognizer

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
    img_size = (args.image_size, args.image_size)
    input_shape = (args.image_size, args.image_size, args.num_channels)

    data = np.load(args.image_path + 'perturbed.npz')

    train_data = data['train_data']
    train_label = data['train_label']
    test_data = data['test_data']
    test_label = data['test_label']
    id_gender = data['id_gender']
    num_good_labels = 2
    num_evil_labels = train_label.shape[1]

    print train_data.shape, train_label.shape
    print test_data.shape, test_label.shape
    print id_gender.shape

    good_used = FaceRecognizer('%s_%d_gender_0' % (args.model_path, img_size),
                               num_good_labels,
                               input_shape)

    good_left = FaceRecognizer('%s_%d_gender_1' % (args.model_path, img_size),
                               num_good_labels,
                               input_shape)

    evil_used = FaceRecognizer('%s_%d_id_0' % (args.model_path, img_size),
                               num_evil_labels,
                               input_shape)

    evil_left = FaceRecognizer('%s_%d_id_1' % (args.model_path, img_size),
                               num_evil_labels,
                               input_shape)

if __name__ == '__main__':
    main()
