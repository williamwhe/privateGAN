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
from sklearn.model_selection import StratifiedKFold

from lfw import get_people_names, read_data, split_dataset, split_indices, preprocess_images, abs_one_to_prediction
from lfw import get_30_people_chunk

VGG_HIDDEN_DIM = 512

class FaceRecognizer:
    """A face recognizing model based on VGG"""
    def __init__(self, restore, num_classes, input_shape, num_channels=3):
        self.model_path = restore
        self.num_channels = num_channels
        self.input_shape = input_shape
        self.num_classes = num_classes

        vgg_notop = VGGFace(include_top=False, input_shape=self.input_shape)

        last_layer = vgg_notop.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)

        # Put two fully-connected layers after it.
        x = Dense(VGG_HIDDEN_DIM, activation='relu', name='fc6')(x)
        x = Dense(VGG_HIDDEN_DIM, activation='relu', name='fc7')(x)

        # Finally, a Dense layer for the output of classes.
        face_probs = Dense(self.num_classes, name='fc8')(x)

        model = Model(vgg_notop.input, face_probs)

        model.load_weights(restore)

        self.model = model

    def predict(self, data):
        """Wrapper function for prediction"""
        return self.model(data)

def train(train_xy,
          test_xy=None,
          save_path=None,
          input_shape=(224, 224, 3),
          batch_size=128,
          num_epochs=25,
          lr=0.001,
          fixed_low_level=False,
          gender=False):

    train_data, train_label = train_xy
    if test_xy:
        test_data, test_label = test_xy

    num_classes = train_label.shape[1]
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

    if fixed_low_level:
        # We make low-level layers untrainable.
        for i in range(len(model.layers) - 3, len(model.layers)):
            model.layers[i].trainable = False

    # A softmax loss is used for training of this network.
    def softmax(correct, predicted):
        """
        The softmax loss function. For more than 2 one-hot encoded classes.
        """
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=correct, logits=predicted)

    print 'Learning rate: %f' % lr

    model.compile(loss=softmax,
                  optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    # Model is trained.
    if save_path:
        # We train one final time to save the model.
        model.fit(train_data, train_label,
                  batch_size=batch_size,
                  epochs=num_epochs,
                  shuffle=True)
        if gender:
            model.save(save_path + '_%d_gender' % input_shape[0])
        else:
            model.save(save_path + '_%d_id' % input_shape[0])
    else:
        # Training with validation data.
        model.fit(train_data, train_label,
                  batch_size=batch_size,
                  validation_data=(test_data, test_label),
                  epochs=num_epochs,
                  shuffle=True)
        # We only return the validation accuracy.
        return model.evaluate(test_data,
                              test_label,
                              batch_size=batch_size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="models/lfw",
                        help="Path to save trained model. e.g.: 'models/lfw'")
    parser.add_argument('--min_num_pics', type=int, default=30,
                        help='Minimum number of pictures for people to be included in data.')
    parser.add_argument('--image_path', type=str, default='./lfw_data/',
                        help='Path to LFW data.')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Size of input images.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size used for training.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--gender', dest='gender', action='store_true',
                        help='Does gender identification.')
    parser.add_argument('--no_gender', dest='gender', action='store_false',
                        help='Does identity recognition.')
    parser.set_defaults(gender=False)
    parser.add_argument('--slice_num', type=int, default=0,
                        help='Which slice to use [0, 1 or 2]')
    parser.add_argument('--fixed_low_level', dest='fixed_low_level', action='store_true',
                        help='With fixed low level layers.')
    parser.add_argument('--no_fixed_low_level', dest='fixed_low_level', action='store_false',
                        help='Without fixed low level layers.')
    parser.set_defaults(fixed_low_level=False)
    args = parser.parse_args()

    img_size = (args.image_size, args.image_size)
    input_shape = (args.image_size, args.image_size, 3)

    imgs, lbls = get_30_people_chunk(args.image_path, args.slice_num, args.gender, img_size)
    print 'Data is read.'

    # names = get_people_names(args.image_path, args.min_num_pics)
    # imgs, identity = read_data(args.image_path,
    #                            names,
    #                            img_size=img_size,
    #                            gender_label=False)

    # if args.gender:
    #     _, gender = read_data(args.image_path,
    #                           names,
    #                           img_size=img_size,
    #                           gender_label=True)
    # if args.gender:
    #     lbls = gender
    # else:
    #     lbls = identity
    # imgs, lbls = read_data(args.image_path, names, img_size=img_size, gender_label=args.gender)
    # numerical_lbls = np.argmax(lbls, axis=1)

    # We calculate the categorical labels. This is used to create K-fold slices.
    lbl_cat = np.argmax(lbls, axis=1)
    skf = StratifiedKFold(n_splits=4)
    slices = skf.split(imgs, lbl_cat)

    print 'Shape of labels:', lbls.shape
    print 'Saving directory: %s' % args.path
    print 'input shape:', input_shape
    print 'batch size:', args.batch_size
    print 'num_epochs:', args.num_epochs

    for train_idx, test_idx in slice:
        train_xy = (imgs[train_idx, :], lbls[train_idx, :])
        test_xy = (imgs[test_idx, :], lbls[test_idx, :])
        score = train(train_xy, test_xy,
                      input_shape=input_shape,
                      batch_size=args.batch_size,
                      num_epochs=args.num_epochs,
                      lr=args.lr,
                      gender=args.gender,
                      fixed_low_level=args.fixed_low_level)
        print score
        np.savez_compressed('score', score=score)
        print score[1]
        break

    # if args.gender:
    #     # We take other images to train a gender identifier.
    #     names = get_people_names(args.image_path, 30)
    #     others = np.setdiff1d(get_people_names(args.image_path), names)
    #     other_imgs, other_lbls = read_data(args.image_path, others,
    #                                        img_size=img_size, gender_label=True)
    #     split_data.train.data = np.concatenate((split_data.train.data, other_imgs))
    #     split_data.train.lbl = np.concatenate((split_data.train.lbl, other_lbls))


    # print split_data.train.data.shape
    # print split_data.train.lbl.shape

    # print split_data.valid.data.shape
    # print split_data.valid.lbl.shape

    # print split_data.test.data.shape
    # print split_data.test.data.shape

    # print 'Preprocessing the images.'
    # for data in [split_data.train.data, split_data.valid.data, split_data.test.data]:
    #     data = preprocess_images(data)

    # train(split_data, args.path,
    #       input_shape=input_shape,
    #       batch_size=args.batch_size,
    #       num_epochs=args.num_epochs,
    #       lr=args.lr,
    #       gender=args.gender)

if __name__ == '__main__':
    main()
