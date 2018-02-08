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
              to_array=True,
              gender_meta=False):

    imgs = []
    lbls = []
    cnt = 0

    if gender_label or gender_meta:
        # We create a dictionary of people's genders.
        genders = pd.read_csv('./lfw_data/gender.csv')
        genders = dict(zip(genders.name.tolist(), genders.gender.tolist()))

    id_gender = []
    for name in os.listdir(img_path):
        if (selected_names is not None) and (name in selected_names):
            print selected_names
            # print '\r%d/%d read.' % (cnt + 1, len(selected_names)),
            folder = os.path.join(img_path, name)
            if gender_meta:
                id_gender.append(genders[name])
            if gender_label:
                if genders.get(name) is None:
                    # print '\tNo gender label found for %s' % name
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
    if gender_meta:
        id_gender = to_categorical(np.array(id_gender))
    lbls = np.array(lbls)
    if one_hot_encoding is True:
        lbls = to_categorical(lbls)
    imgs = np.concatenate(imgs)

    if gender_meta:
        return imgs, lbls, id_gender
    else:
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

def abs_one_to_prediction(imgs):
    # from (-1, 1) to (0, 255)
    return preprocess_images((imgs + 1) * 127.5, version=1)

def print_ready(img):
    return image.array_to_img(postprocess_images(img, version=1))

def preprocess_images(x, version=1):
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
    if len(lbls.shape) > 1:
        train_data, train_lbl = data[trn, :], lbls[trn, :]
        test_data, test_lbl = data[tst, :], lbls[tst, :]
        valid_data, valid_lbl = data[val, :], lbls[val, :]

    return Namespace(train=(Namespace(data=train_data, lbl=train_lbl)),
                     valid=(Namespace(data=valid_data, lbl=valid_lbl)),
                     test=(Namespace(data=test_data, lbl=test_lbl)))

def balance_dataset(data, label, ratio=2):
    min_pic_person = int(np.min(np.sum(label, axis=0)))
    print 'There are %d minimum pictures per person.' % min_pic_person
    max_pic_person = int(ratio * min_pic_person)
    print 'Max would be %d' % max_pic_person

    class_indices = dict()
    for index, cls in enumerate(np.argmax(label, axis=1)):
        if class_indices.get(cls) is None:
            class_indices[cls] = [index]
        else:
            class_indices[cls].append(index)

    final_indices = []
    for cls, indices in class_indices.items():
        if len(indices) < max_pic_person:
            final_indices.extend(indices)
        else:
            final_indices.extend(
                np.random.choice(indices, max_pic_person, replace=False).tolist())

    return data[final_indices, :], label[final_indices, :]

def get_gender(names, gender_path='lfw_data/gender.csv'):
    if os.path.exists(gender_path) is False:
        raise ValueError('File %s does not exist.' % gender_path)

    genders = pd.read_csv(gender_path)
    names = pd.DataFrame(data=names, columns=['name'])

    return pd.merge(genders, names, right_on='name', left_on='name').gender.values

def get_30_people_chunk(image_path,
                        chunk_number,
                        gender_label=False,
                        img_size=(224, 224),
                        gender_meta=False):

    if chunk_number < 0 or chunk_number >= 3:
        raise ValueError('chunk_number(%d) should be between 0 and 3' % chunk_number)
    names = get_people_names(image_path, 30)
    print '\n'.join(names)
    if gender_meta:
        imgs, lbls, id_gender = read_data(image_path, names, img_size=img_size,
                                          gender_label=gender_label, gender_meta=True)
    else:
        imgs, lbls = read_data(image_path, names, img_size=img_size, gender_label=gender_label)

    indices = split_indices(lbls)
    imgs = imgs[indices[chunk_number], :]
    imgs = imgs / 255.0
    lbls = lbls[indices[chunk_number], :]

    if gender_meta:
        return imgs, lbls, id_gender
    # if gender_meta is False:
    return imgs, lbls

def main():
    image_path = 'lfw_data/'
    img_size = (224, 224)

    imgs, lbls = get_30_people_chunk(image_path, 1)
    print imgs.shape
    print lbls.shape

if __name__ == '__main__':
    main()
