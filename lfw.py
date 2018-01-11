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

def read_data(img_path,
              selected_names,
              img_size=(224, 224),
              one_hot_encoding=True,
              gender_label=False,
              to_array=True):

    imgs = []
    lbls = []
    cnt = 0

    if gender_label:
        # We create a dictionary of people's genders.
        genders = pd.read_csv('./lfw_data/gender.csv')
        genders = dict(zip(genders.name.tolist(), genders.gender.tolist()))

    for name in os.listdir(img_path):
        if (selected_names is not None) and (name in selected_names):
            print '\r%d/%d read.' % (cnt + 1, len(selected_names)),
            folder = os.path.join(img_path, name)
            if gender_label:
                if genders.get(name) is None:
                    print '\tNo gender label found for %s' % name
                    continue
                is_male = genders[name]
            if os.path.isdir(folder):
                for fn in os.listdir(folder):
                    if gender_label:
                        lbls.append(is_male)
                    else:
                        lbls.append(cnt)
                    img = image.load_img(os.path.join(folder, fn), target_size=img_size)
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    imgs.append(x)
            cnt += 1
    print ''
    lbls = np.array(lbls)
    if gender_label is False and one_hot_encoding is True:
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

def get_gender(names, gender_path='lfw_data/gender.csv'):
    if os.path.exists(gender_path) is False:
        raise ValueError('File %s does not exist.' % gender_path)

    genders = pd.read_csv(gender_path)
    names = pd.DataFrame(data=names, columns=['name'])

    return pd.merge(genders, names, right_on='name', left_on='name').gender.values


def main():
    image_path = 'lfw_data/'
    img_size = (224, 224)

    names = get_people_names(image_path, 30)
    imgs, identity = read_data(image_path,
                               names,
                               img_size=img_size,
                               gender_label=False)
    _, gender = read_data(image_path,
                          names,
                          img_size=img_size,
                          gender_label=True)

    train, _, _ = split_indices(identity)

    others = np.setdiff1d(get_people_names(image_path), names)
    other_imgs, other_gender = read_data(image_path,
                                         others,
                                         img_size=img_size,
                                         gender_label=True)

    print other_imgs.shape
    print imgs[train, :].shape
    # exit()
    train_data = np.concatenate((imgs[train, :], other_imgs))
    train_gender = np.concatenate((gender[train], other_gender))

    print train_data.shape
    print train_gender.shape

if __name__ == '__main__':
    main()
