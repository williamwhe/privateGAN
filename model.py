import pdb
from sklearn import metrics
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# import tensorflow as tf
import math
from sklearn.ensemble import VotingClassifier
import cPickle
from sklearn.externals import joblib
import tensorflow as tf
import prettytensor as pt
import os
class Discriminator(object):
    """
    The basic class for discriminator
    """
    def train(self, X, y):
        """ train the model
        :param X: the data matrix
        :param y: the label
        """
        raise NotImplementedError("Abstract method")

    def predict(self, X):
        """ predict the label of X
        :param X: the data matrix
        :return: the label of X
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """ evaluate the classification performance of X with respect of y
        :param X: the test data
        :param y: the ground truth label of X
        :return: a dict of performance scores
        """
        return self._score(y, self.model.predict_proba(X)[:, 1])

    def predict_proba(self, X):
        return self.model.predict_proba(X)


    def _score(self, true_label, predicted_prob):
        """ calculate the performance score for binary calssification
        :param true_label: the ground truth score
        :param predicted_label: the predicted probability
        :return: a dict of scores
        """
        score_dict = dict()
        score_dict['AUC'] = metrics.roc_auc_score(true_label, predicted_prob)
        predicted_label = [0 if prob < 0.5 else 1 for prob in predicted_prob]
        score_dict['Accuracy'] = metrics.accuracy_score(true_label, predicted_label)
        cm = metrics.confusion_matrix(true_label, predicted_label)
        score_dict['Confusion Matrix'] = cm
        #TPR = TP /(TP + FN)
        score_dict['TPR'] = cm[1, 1] / float(cm[1, 0] + cm[1, 1])
        score_dict['FPR'] = cm[0, 1] / float(cm[0, 0] + cm[0, 1])
        # score_dict['FNR'] = cm[0, 0] / float(cm[0,0] + cm[0, 1])
        #FNR = FN/(TP + FN)
        score_dict['FNR'] = cm[1, 0] / float(cm[1,0] + cm[1, 1])
        return score_dict

    def save(self, fname):
        with open(fname, 'wb') as fid:
            cPickle.dump(self.model, fid)

    def load(self, fname):
        with open(fname, 'rb') as fid:
            self.model = cPickle.load(fid)

    def getName(self):
        return self.name

class RandomForrest(Discriminator):
    """
    using RF as the discriminator
    """

    def __init__(self, num_trees=100, num_threads=25):
        self.num_trees = num_trees
        self.num_threads = num_threads
        self.model = None
        self.name = "rf"
    def train(self, X, y):
        self.model = RandomForestClassifier(n_estimators=self.num_trees, n_jobs=self.num_threads)
        self.model.fit(X, y)


class LR(Discriminator):
    """
    using RF as the discriminator
    """

    def __init__(self):
        self.model = None
        self.name = "lr"
    def train(self, X, y):
        self.model = LogisticRegression()
        self.model.fit(X, y)


class DT(Discriminator):
    """
    using RF as the discriminator
    """

    def __init__(self):
        self.model = None
        self.name = "dt"

    def train(self, X, y):
        self.model = DecisionTreeClassifier()
        self.model.fit(X, y)


class NB(Discriminator):
    """
    using RF as the discriminator
    """

    def __init__(self):
        self.model = None
        self.name = "nb"

    def train(self, X, y):
        self.model = BernoulliNB()
        self.model.fit(X, y)


class MLP(Discriminator):
    """
    using RF as the discriminator
    """

    def __init__(self, layers=(100,)):
        self.layers = layers
        self.model = None
        self.name = "mlp"
    def train(self, X, y):
        self.model = MLPClassifier(hidden_layer_sizes=self.layers, activation = 'relu', early_stopping=True)
        self.model.fit(X, y)
    def predict(self, X, y):
        y_pred = self.model.predict(X)
        y_binary = np.zeros( y.shape[0] ) 
        collision_id =np.where(  np.sum( abs(y_pred - y ), axis = 1) > 0)
        y_binary[collision_id]= 1
        return y_binary

class SVM(Discriminator):
    """
    using RF as the discriminator
    """

    def __init__(self):
        self.model = None
        self.name = "svm"
    def train(self, X, y):
        self.model = SVC(probability=True)
        self.model.fit(X, y)


class KNN(Discriminator):
    """
    using RF as the discriminator
    """

    def __init__(self):
        self.model = None
        self.name = "knn"
    def train(self, X, y):
        self.model = KNeighborsClassifier(n_jobs=10)
        self.model.fit(X, y)


class VOTE(Discriminator):
    """
    using RF as the discriminator
    """

    def __init__(self):
        self.model = None
        self.name = "vote"
    def train(self, X, y):
        clf1 = LogisticRegression()
        clf2 = RandomForestClassifier(n_estimators=100, n_jobs=10)
        clf3 = BernoulliNB()
        clf4 = DecisionTreeClassifier()
        clf5 = MLPClassifier(early_stopping=True)
        clf6 = SVC(probability=True)
        clf7 = KNeighborsClassifier(n_jobs=10)
        self.model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('bnb', clf3), ('dt', clf4),
                                                  ('mlp', clf5), ('svm', clf6), ('knn', clf7)], voting='soft')
        self.model.fit(X, y)

class CNN():

    def __init__(self, opt, sess):
        self.opts = opt

        self.sess = sess
        self.images = tf.placeholder(tf.float32, \
            [None,  self.opts.img_dim, self.opts.img_dim, self.opts.input_c_dim], name='image')

        self.y_true = tf.placeholder(tf.float32, shape=[None, self.opts.label_dim], name='y_true')

        y_true_cls = tf.argmax(self.y_true, dimension=1)

        _, self.loss = self.create_network(training=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)  
        y_pred, _ = self.create_network(training=False)
        y_pred_cls = tf.argmax(y_pred, dimension=1)
        self.y_pred_cls = y_pred_cls
        self.y_pred = y_pred
        y_pred_cls = tf.argmax(y_pred, dimension=1)
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        self.y_prediction = tf.not_equal(y_pred_cls, y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        net_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='network')
        self.saver = tf.train.Saver( net_vars, max_to_keep =1)
        self.sess.run( tf.variables_initializer(net_vars) )
        # self.sess.run(tf.global_variables_initializer())


    def predict(self, data,  label):
        feed = {self.images : data, self.y_true : label }
        y_prediction = self.sess.run( self.y_prediction, feed)
        return y_prediction

    def create_network(self, training):
        # Wrap the neural network in the scope named 'network'.
        # Create new variables during training, and re-use during testing.
        with tf.variable_scope('network', reuse=not training):
            # Just rename the input placeholder variable for convenience.
            # Create TensorFlow graph for the main processing.
            y_pred, loss = self.main_network(images=self.images, training=training)

        return y_pred, loss


    def main_network(self, images, training):
        # Wrap the input images as a Pretty Tensor object.
        x_pretty = pt.wrap(images)

        # Pretty Tensor uses special numbers to distinguish between
        # the training and testing phases.
        if training:
            phase = pt.Phase.train
        else:
            phase = pt.Phase.infer

        # Create the convolutional neural network using Pretty Tensor.
        # It is very similar to the previous tutorials, except
        # the use of so-called batch-normalization in the first layer.
        with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
            y_pred, loss = x_pretty.\
                conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).\
                max_pool(kernel=2, stride=2).\
                conv2d(kernel=5, depth=64, name='layer_conv2', batch_normalize=True).\
                max_pool(kernel=2, stride=2).\
                flatten().\
                fully_connected(size=256, name='layer_fc1').\
                fully_connected(size=128, name='layer_fc2').\
                softmax_classifier(num_classes=self.opts.label_dim, labels=self.y_true)

        return y_pred, loss

    def load(self, checkpoint_path):
        self.saver.restore(self.sess, checkpoint_path)
    


def setup(opt, sess):
    model_path_name = "%s/%s_model.ckpt" %(opt.load_checkpoint_path, opt.model_name.lower()) 
    # check compatibility if training is continued from previously saved model
    print 'model path: ', model_path_name

    if opt.model_name.upper() == 'MLP':
        model = MLP()
        model.load( model_path_name)
        return model
    elif opt.model_name.upper() == 'CNN':
        model =  CNN(opt, sess)
        model.load(model_path_name)
        return model
    else:
        raise Exception("Caption model not supported: {}".format(opt.model_name))
