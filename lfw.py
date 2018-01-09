"""This is a module to read and manipulate LFW data"""
# pylint: disable=C0103, C0111
from __future__ import division

import os
from argparse import Namespace

import pandas as pd
import numpy as np
from keras.preprocessing import image
from keras.utils import to_categorical
import keras.backend as K
import keras_vggface

def get_people_names(img_path, min_num_pics=1, record_file_name='lfw-names.txt'):
    """"Returns name of people with more than `min_num_pics` pictures."""
    if os.path.isfile(os.path.join(img_path, record_file_name)):
        recs = pd.read_csv(os.path.join(img_path, record_file_name),
                           header=None, names=['name', 'num'], delimiter='\t')
    else:
        raise ValueError('LFW name records are not available at %s' % (
            os.path.join(img_path, record_file_name)))

    return recs[recs.num >= min_num_pics].name.values

def read_data(img_path, selected_names=None, img_size=(182, 182), categorical=True):
    imgs = []
    lbls = []
    cnt = 0

    for folder in os.listdir(img_path):
        if (selected_names is not None) and (folder in selected_names):
            folder = os.path.join(img_path, folder)
            if os.path.isdir(folder):
                for fn in os.listdir(folder):
                    lbls.append(cnt)
                    img = image.load_img(os.path.join(folder, fn), target_size=img_size)
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    imgs.append(x)
            cnt += 1
    lbls = np.array(lbls)
    if categorical:
        lbls = to_categorical(lbls)
    imgs = np.concatenate(imgs)

    return imgs, lbls

def postprocess_images(x, data_format=None, version=1):
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x[:, 0, :, :] += 93.5940
            x[:, 1, :, :] += 104.7624
            x[:, 2, :, :] += 129.1863
            x = x[:, ::-1, ...]
        else:
            x[..., 0] += 93.5940
            x[..., 1] += 104.7624
            x[..., 2] += 129.1863
            x = x[..., ::-1]

    elif version == 2:
        if data_format == 'channels_first':
            x[:, 0, :, :] += 91.4953
            x[:, 1, :, :] += 103.8827
            x[:, 2, :, :] += 131.0912
            x = x[:, ::-1, ...]
        else:
            x[..., 0] += 91.4953
            x[..., 1] += 103.8827
            x[..., 2] += 131.0912
            x = x[..., ::-1]
    else:
        raise NotImplementedError

    return x

def preprocess_images(x, version=1):
    print x.shape
    return keras_vggface.utils.preprocess_input(x, version=version)

def split_indices(lbls, training_portion=.4, validation_portion=.4):
    indices = dict()
    numerical_lbls = np.argmax(lbls, axis=1)
    for (idx, lbl) in enumerate(numerical_lbls):
        if indices.get(lbl) is None:
            indices[lbl] = [idx]
        else:
            indices[lbl].append(idx)
    trn = []
    val = []
    tst = []
    for lbl, samples in indices.items():
        num_samples = len(samples)
        num_trn = int(training_portion * num_samples)
        num_val = int(validation_portion * num_samples)
        for split_idx, dest_chunk in zip(np.split(samples, [num_trn, num_trn + num_val]),
                                         [trn, val, tst]):
            dest_chunk.extend(split_idx)

    return trn, val, tst

def split_dataset(data, lbls, indices):
    trn, val, tst = indices
    train_data, train_lbl = data[trn, :], lbls[trn, :]
    test_data, test_lbl = data[tst, :], lbls[tst, :]
    valid_data, valid_lbl = data[val, :], lbls[val, :]

    return Namespace(train=(Namespace(data=train_data, lbl=train_lbl)),
                     valid=(Namespace(data=valid_data, lbl=valid_lbl)),
                     test=(Namespace(data=test_data, lbl=test_lbl)))

def main():
    print "Testing the input method."
    names = ['Aaron_Eckhart', 'Aaron_Tippin']
    imgs, lbls = read_data('lfw_data/', names)
    print imgs.shape
    print lbls

if __name__ == '__main__':
    main()

