"""Here are the modules for face recognition"""
from __future__ import division

import time
import os
import argparse
import cPickle as pickle

import numpy as np
import keras_vggface
from keras_vggface.vggface import VGGFace
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input

from lfw import get_people_names, read_data

def get_or_create_bottleneck(vgg_model, data, bottleneck_dir='./bottleneck', batch_size=100):
    """Creates or, if previously created and saved, loads the bottleneck for the data"""
    start_time = time.time()
    if os.path.exists(os.path.join(bottleneck_dir, 'bottleneck.pkl')):
        print 'hello :)'
        with open(os.path.join(bottleneck_dir, 'bottleneck.pkl')) as bn_file:
            features = pickle.load(bn_file)
    else:
        print 'No save file detected. Creating features and saving them at %s' % \
            os.path.join(bottleneck_dir, 'bottleneck.pkl')
        num_data = data.shape[0]
        features = []
        for i in range(0, num_data, batch_size):
            print '\r%d/%d' % (min(i + batch_size, num_data), num_data),
            features.append(vgg_model.predict(data[i:min(i + batch_size, num_data), :]))
        print ''
        features = np.concatenate(features)
        with open(os.path.join(bottleneck_dir, 'bottleneck.pkl'), 'w') as bn_file:
            pickle.dump(features, bn_file)
    print 'Elapsed time: %.2f' % (time.time() - start_time)
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_num_pics', type=int, default=20,
                        help='Minimum number of pictures for people to be included in data.')
    parser.add_argument('--bottleneck_dir', type=str, deafult='./bottleneck',
                        help='Path saved bottleneck data.')
    parser.add_argument('--bottleneck_batch_size', type=int, default=100,
                        help='Size of the batch for creating bottleneck.')
    parser.add_argument('--image_path', type=str, default='./lfw/',
                        help='Path to LFW data.')
    parser.add_argument('--image_size', type=int, default=182,
                        help='Size of input images.')
    args = parser.parse_args()

    vgg_features = VGGFace(include_top=False,
                           input_shape=(args.image_size, args.image_size, 3),
                           pooling='avg')
    names = get_people_names(args.image_path, args.min_num_pics)
    print 'Retrieved qualifying names.'
    imgs, lbls = read_data(args.image_path, names)
    print 'Read data.'
    features = get_or_create_bottleneck(vgg_features, imgs, batch_size=20)
    print 'Features calculated/loaded.'

if __name__ == '__main__':
    main()
