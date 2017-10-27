import pdb
import numpy as np
import scipy.io as sio
import random



def odd_even_labels(labels, one_hot=True):
    """
    Turns digit classes to odd[1]/even[0].
    Returns hot-one encoded labels.
    """
    if one_hot is True:
        labels = np.argmax(labels, axis=1)

    num_samples = labels.shape[0]
    new_labels = np.zeros((num_samples, 2))

    for mod_2 in range(2):
        # new_labels[even, 0] = 1
        # new_labels[odd, 1] = 1
        new_labels[labels % 2 == mod_2, mod_2] = 1

    return new_labels

class Dataset2:
    """
    Batch loader class for datasets.
    """

    def __init__(self, data, label, source_label=None, targeted=False):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._label = label
        self._num_examples = data.shape[0]
        self._num_labels = label.shape[1]
        self._targeted = targeted
            # Used to retrieve negative samples.
        self._negative_example_list = [
            np.array([]) for _ in range(self._num_labels)]
        self._negative_magnitude = [
            np.array([]) for _ in range(self._num_labels)]
        # Finding  the class indices, from the one-hot format.
        label_cls = np.argmax(self._label, axis=1)
        # A list of examples for each class
        examples = [[] for _ in range(self._num_labels)]
        for i in range(self._num_examples):
            label_idx = label_cls[i]
            examples[label_idx].append(self._data[i])

        # We turn the gathered examples to numpy arrays.
        self._sort_data = [np.array(examples[i])
                        for i in range(self._num_labels)]

        # We find the number of examples for each class.
        self._sort_len = [self._sort_data[i].shape[0]
                        for i in range(self._num_labels)]

        if self._targeted is True:
            # We make data to be only instances of 'source_label'
            self._data = self._sort_data[source_label]
            # Create one-hot labels.
            self._num_examples = self._data.shape[0]
            self._label = np.zeros((self._num_examples, self._num_labels))
            self._label[:, source_label] = 1
        else: # targeted == False, we use all data.
            idx = np.arange(0, self._num_examples)
            np.random.shuffle(idx)  # shuffle indices.
            self._data = self._data[idx]  # shuffled data samples.
            self._label = self._label[idx] # shuffled labels.

    @property
    def data(self):
        return self._data

    def next_batch(self, batch_size, negative=False,
                   priority=False, shuffle=True):
        """
        Params:
            batch_size (int): size of the batch.
            negative (bool): If True, negative examples are returned too.
            priority (bool):
            shuffle (bool): If True, data is randomly shuffled.
        Returns:
            next batch of data.
        """
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            # np.random.shuffle(idx)  # shuffle indexe
            self._data = self._data[idx]  # get list of `num` random samples
            self._label = self._label[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            label_rest_part = self._label[start: self._num_examples]

            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            # np.random.shuffle(idx0)  # shuffle indexes
            self._data = self._data[idx0]  # get list of `num` random samples
            self._label = self._label[idx0]

            start = 0
            # avoid #sample != integar x batch_size
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            label_new_part = self._label[start:end]
            data = np.concatenate(
                (data_rest_part, data_new_part), axis=0)
            label = np.concatenate(
                (label_rest_part, label_new_part), axis=0)
            label_cls = np.argmax(label, axis=1)
            target = []
            target_label = []
            for i in range(batch_size):
                label_cls_id = label_cls[i]
                target.append(random.choice(self._sort_data[label_cls_id]))

            if negative:
                neg_data, neg_label = self.get_negative(
                    batch_size, label, priority)
                return data, label, neg_data, neg_label
            else:
                return data, label, np.array(target)
        else: # start + batch_size <= self._num_examples.
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            data, label = self._data[start:end], self._label[start:end]
            label_cls = np.argmax(label, axis=1)
            target = []
            for i in range(batch_size):
                label_cls_id = label_cls[i]
                target.append(random.choice(self._sort_data[label_cls_id]))
            if negative: # negative examples are needed.
                neg_data, neg_label = self.get_negative(
                    batch_size, label, priority)
                return data, label, neg_data, neg_label
            else:
                return data, label, np.array(target)

    # random generate negative samples
    # may take a long time
    # priority == Ture means that we select the negative samples from the
    # labeled negative samples at first

    def get_negative(self, batch_size, labels, priority=False, threshold=2):
        """
        Randomly generates negative samples. This might take too long.
        Params:
            batch_size (int): Size of the batch.
            labels (list):
            priority (bool): If True, we select negative samples from labeled samples first.
            threshold (float):
        """
        if priority:
            neg_datas = []
            neg_labels = []
            label_cls = np.argmax(labels, axis=1)
            for i in range(len(label_cls)):
                label_idx = label_cls[i]
                label_idx_len = len(self._negative_example_list[label_idx])
                if label_idx_len > threshold and random.random() > 0.1:
                    neg_datas.append(
                        random.choice(
                            self._negative_example_list[label_idx]))
                    neg_label = np.zeros(self._num_labels)
                    neg_label[label_idx] = 1
                    neg_labels.append(neg_label)
                else:
                    while True:
                        fake_id = random.randint(0, self._num_examples - 1)
                        if not np.argmax(self._label[fake_id]) == label_idx:
                            break
                    neg_datas.append(self._data[fake_id, :])
                    neg_labels.append(self._label[fake_id, :])
            neg_datas = np.array(neg_datas)
            neg_labels = np.array(neg_labels)
            return neg_datas, neg_labels
        else:
            fake_ids = np.random.randint(self._num_examples, size=batch_size)
            self._label[fake_ids]
            collision_flag = range(batch_size)
            while True:
                collision_flag = \
                    np.array(
                        np.where(np.sum(abs(self._label[fake_ids] - labels), axis=1) == 0)[0])
                if len(collision_flag) == 0:
                    break
                fake_ids[collision_flag] = np.random.randint(
                    self._num_examples, size=len(collision_flag))
            return self._data[fake_ids], self._label[fake_ids]

    def insert_negative_sample(self, data, labels, magnitude, threshold=0):
        label_cls = np.argmax(labels, axis=1)
        for i, label_i in enumerate(label_cls):
            if self._negative_example_list[label_i].shape[0] == 0:
                self._negative_example_list[label_i] = data[i, :]
                self._negative_magnitude[label_i] = magnitude[i]

            self._negative_example_list[label_i] = \
                np.vstack([self._negative_example_list[label_i], data[i, :]])

            self._negative_magnitude[label_i] = \
                np.vstack([self._negative_magnitude[label_i], magnitude[i]])

            # self._negative_example_list[label_i] = np.concatenate( ( self._negative_example_list[label_i], data[i, :] ), axis = 0)

            # self._negative_magnitude[label_i] = np.concatenate( ( self._negative_magnitude[label_i], magnitude[i, :]), axis = 0)
            if len(self._negative_example_list[label_i]) > (threshold + 300):
                # remove ... only keep the most like example
                sort_idx = np.argsort(self._negative_magnitude[label_i][:, 0])
                self._negative_example_list[label_i] = \
                    self._negative_example_list[label_i][sort_idx[-threshold:], :]
                self._negative_magnitude[label_i] = \
                    self._negative_magnitude[label_i][sort_idx[-threshold:], :]

    def save_negative_sample(self, fname):
        sio.savemat(fname, {
            "negative": np.array(self._negative_example_list),
            "magnitude": np.array(self._negative_magnitude)
        })

    def load_negative_sample(self, fname):
        d = sio.loadmat(fname)
        self._negative_example_list = d["negative"]
        self._negative_magnitude = d["magnitude"]

    def get_image_by_index(self, label, b_size=1):
        new_label = np.argmax(self._label, axis=1)
        idxs = np.where(new_label == label)[0]
        idxs = np.random.shuffle(idxs)[:b_size]
        return self._data[idxs]

    def get_all_images(self, batch_size=16):
        imgs = self._data[0:batch_size]
        labels = self._label[0:batch_size]
        # label_dim = self._label.shape[1]
        # new_label = np.argmax(self._label, axis = 1)
        # imgs = []
        # labels = []
        # for i in range(batch_size):
        #     i = i % label_dim
        #     # pdb.set_trace()
        #     idx = random.choice( np.where(new_label == i)[0] )
        #     imgs.append(self._data[idx, :])
        #     labels.append(self._label[idx, :])
        # imgs = np.array(imgs)
        # labels = np.array(labels)
        return imgs, labels

    # def get_all_images(self, batch_size = 16):
    #     label_dim = self._label.shape[1]
    #     new_label = np.argmax(self._label, axis = 1)
    #     imgs = []
    #     labels = []
    #     for i in range(batch_size):
    #         i = i % label_dim
    #         # pdb.set_trace()
    #         idx = random.choice( np.where(new_label == i)[0] )
    #         imgs.append(self._data[idx, :])
    #         labels.append(self._label[idx, :])
    #     imgs = np.array(imgs)
    #     labels = np.array(labels)
    #     return imgs, labels
