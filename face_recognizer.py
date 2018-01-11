"""Here are the modules for face recognition"""
# pylint: disable=C0103, C0111
from __future__ import division

import time
import os
import argparse
import cPickle as pickle

import numpy as np
import tensorflow as tf
import keras_vggface
from keras_vggface.vggface import VGGFace
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras.optimizers import Adam, SGD

from lfw import get_people_names, read_data, split_dataset, split_indices, preprocess_images

VGG_HIDDEN_DIM = 512

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


def train(split_data,
          save_path=None,
          input_shape=(224, 224, 3),
          batch_size=128,
          num_epochs=100,
          gender=False):

    num_classes = split_data.train.lbl.shape[1]
    num_images = split_data.train.data.shape[0]
    vgg_notop = VGGFace(include_top=False, input_shape=input_shape)

    # We take the output of the last MaxPooling layer.
    last_layer = vgg_notop.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)

    # Put two fully-connected layers after it.
    x = Dense(VGG_HIDDEN_DIM, activation='relu', name='fc6')(x)
    x = Dense(VGG_HIDDEN_DIM, activation='relu', name='fc7')(x)

    # Finally, a Dense layer for the output of classes.
    face_probs = Dense(num_classes, name='fc8')(x)

    model = Model(vgg_notop.input, face_probs)

    # We make all other layers UNTRAINABLE.
    for i in range(len(model.layers) - 3, len(model.layers)):
        model.layers[i].trainable = False

    # A softmax loss is used for training of this network.
    def softmax(correct, predicted):
        """
        The softmax loss function. For more than 2 one-hot encoded classes.
        """
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=correct, logits=predicted)

    def sigmoid(correct, predicted):
        """
        The sigmoid loss function. For 2 classes.
        """
        return tf.nn.sigmoid_cross_entropy_with_logits(
            labels=correct, logits=predicted)


    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    if gender:
        loss_fn = sigmoid
    else:
        loss_fn = softmax
    model.compile(loss=loss_fn,
                  optimizer=sgd,
                  metrics=['accuracy'])

    # Model is trained.
    model.fit(split_data.train.data, split_data.train.lbl,
              batch_size=batch_size,
              validation_data=(split_data.valid.data, split_data.valid.lbl),
              epochs=num_epochs,
              shuffle=True)

    if save_path:
        if gender:
            model.save(save_path + '_gender')
        else:
            model.save(save_path + '_id')

    # We also test on the left-out test data.
    print model.evaluate(split_data.test.data,
                         split_data.test.lbl,
                         batch_size=batch_size)

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="models/lfw",
                        help="Path to save trained model. e.g.: 'models/lfw'")
    parser.add_argument('--min_num_pics', type=int, default=30,
                        help='Minimum number of pictures for people to be included in data.')
    parser.add_argument('--bottleneck_dir', type=str, default='./bottleneck',
                        help='Path saved bottleneck data.')
    parser.add_argument('--bottleneck_batch_size', type=int, default=100,
                        help='Size of the batch for creating bottleneck.')
    parser.add_argument('--image_path', type=str, default='./lfw_data/',
                        help='Path to LFW data.')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Size of input images.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size used for training.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--gender', dest='gender', action='store_true',
                        help='Does gender identification.')
    parser.add_argument('--no-gender', dest='gender', action='store_false',
                        help='Does identity recognition.')
    parser.set_defaults(gender=False)
    args = parser.parse_args()

    img_size = (args.image_size, args.image_size)
    input_shape = (args.image_size, args.image_size, 3)

    names = get_people_names(args.image_path, args.min_num_pics)
    imgs, identity = read_data(args.image_path,
                               names,
                               img_size=img_size,
                               gender_label=False)

    if args.gender:
        _, gender = read_data(args.image_path,
                              names,
                              img_size=img_size,
                              gender_label=True)
    if args.gender:
        lbls = gender
    else:
        lbls = identity
    # imgs, lbls = read_data(args.image_path, names, img_size=img_size, gender_label=args.gender)
    # numerical_lbls = np.argmax(lbls, axis=1)
    print 'Data is read.'

    # We split the data based on identity
    chunk_indices = split_indices(identity)
    split_data = split_dataset(imgs, lbls, chunk_indices)

    print 'Shape of labels:', lbls.shape

    if args.gender:
        # We take other images to train a gender identifier.
        others = np.setdiff1d(get_people_names(args.image_path), names)
        other_imgs, other_lbls = read_data(args.image_path, others,
                                           img_size=img_size, gender_label=True)
        split_data.train.data = np.concatenate((split_data.train.data, other_imgs))
        split_data.train.lbl = np.concatenate((split_data.train.lbl, other_lbls))

    print split_data.train.data.shape
    print split_data.train.lbl.shape

    print split_data.valid.data.shape
    print split_data.valid.lbl.shape

    print split_data.test.data.shape
    print split_data.test.data.shape

    print 'Preprocessing the images.'
    for data in [split_data.train.data, split_data.valid.data, split_data.test.data]:
        data = preprocess_images(data)

    print 'Number of images %d' % split_data.train.data.shape[0]
    print 'Number of classes %d' % split_data.train.lbl.shape[1]
    print 'Saving directory: %s' % args.path

    print 'input shape:', input_shape
    print 'batch size:', args.batch_size
    print 'num_epochs:', args.num_epochs
    train(split_data, args.path,
          input_shape=input_shape,
          batch_size=args.batch_size,
          num_epochs=args.num_epochs,
          gender=args.gender)


if __name__ == '__main__':
    main()
