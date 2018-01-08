"""This is a module to read and manipulate LFW data"""
# pylint: disable=C0103, C0111
from __future__ import division

import os
import pandas as pd
import numpy as np
from keras.preprocessing import image
from keras.utils import to_categorical

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

def main():
    print "Testing the input method."
    names = ['Aaron_Eckhart', 'Aaron_Tippin']
    imgs, lbls = read_data('../facenet/data/aligned_images/', names)
    print imgs.shape
    print lbls

if __name__ == '__main__':
    main()
