import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from ops import *
import os
import time
import scipy.io as sio
import random
import pdb
import opts
# from utils import plot
from utils import save_images
from advgan import advGAN
import cifar10
from Dataset2 import Dataset2, odd_even_labels

from setup_mnist import MNIST, MNISTModel, MNISTModel2, MNISTModel3, OddEvenMNIST
from setup_cifar import CIFARModel, CIFARModel2, CIFARModel3

tag_num = 0

def train():
    flatten_flag = True  # the output of G need to flatten or not?
    opt = opts.parse_opt()
    opt.input_data = "MNIST"
    # mapping [0,1] -> [-1,1]
    # load data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_data = mnist.train.images * 2.0 - 1.0
    train_label = mnist.train.labels

    test_data = mnist.test.images * 2.0 - 1.0
    test_label = mnist.test.labels

    x_dim = train_data.shape[1]
    y_dim = train_label.shape[1]

    opt.input_c_dim = 1
    opt.output_c_dim = 1
    opt.input_dim = x_dim
    opt.label_dim = y_dim

    batch_size = opt.batch_size

    NUM_THREADS = 2
    tf_config = tf.ConfigProto()
    tf_config.intra_op_parallelism_threads = NUM_THREADS
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        # Initialize the variables, and restore the variables form checkpoint if there is.
        # and initialize the writer
        iteration = 0
        # sample_matrix = []
        # source_matrix = []
        # magnitude_matrix = []
        # min_adv_accuracys = []
        # corsp_magnitudes = []
        #initial whitebox model
        # model_store = opt.model_restore
        # whitebox_model = MNISTModel(model_store, sess)

        print '\tRetrieving evil model from "%s"' % opt.evil_model_path
        evil_model = MNISTModel(opt.evil_model_path)
        print '\tRetrieving good model from "%s"' % opt.good_model_path
        good_model = OddEvenMNIST(opt.good_model_path)
        # exit()
        # model = advGAN(whitebox_model, model_store, opt, sess)
        model = advGAN(good_model, evil_model, opt, sess)

        iteration = 0
        min_adv_accuracy = 10e10
        max_accuracy_diff = -np.inf
        # if opt.is_advGAN:
        #     source_label = opt.s_l
        #     target_label = opt.t_l
        #     if source_label == target_label:
        #         raise ValueError("source and target labels are equal.")
        #     print "source: {}, target: {}.".format(source_label, target_label)

        writer = tf.summary.FileWriter("logs", sess.graph)
        loader = Dataset2(train_data, train_label)
        print 'Training data loaded.'
        # model_num = 0

        # best_show_samples_magintude = []
        # best_show_samples = []
        # best_show_source_imgs = []
        # best_show_idxs = []

        acc_file = open('acc.txt', 'w')
        loss_file = open('loss.txt', 'w')

        save_anything = False

        while iteration < 250:
            # this function returns (data, label, np.array(target)).
            data = loader.next_batch(batch_size, negative=False)
            feed_data, evil_labels, real_data = loader.next_batch(
                batch_size, negative=False)
            good_labels = odd_even_labels(evil_labels)

            # if opt.is_advGAN is True:
            #     labels = np.zeros_like(data[1])
            #     labels[:, target_label] = 1
            # else: # opts.is_advGAN is False.
            #     labels = data[1]

            # feed = {}
            # if opt.is_advGAN is True:
            #     # This is the setting used for advGAN.
            #     feed = {
            #         model.source: data[0],
            #         model.labels: labels,
            #         model.target: data[2]
            #     }
            # else: # opt.is_advGAN == False. Using privateGAN.
            #     feed = {
            #         model.source: data[0],
            #         model.labels: data[1],
            #         model.target: data[0]
            #     }

            feed = {
                model.source: feed_data,
                model.target: real_data,
                model.good_labels: good_labels,
                model.evil_labels: evil_labels
            }

            summary_str, G_loss, pre_G_loss, adv_G_loss, good_fn_loss, \
                evil_fn_loss, hinge_loss, _ = sess.run([
                    model.g_loss_add_adv_merge_sum,
                    model.G_loss_add_adv,
                    model.pre_G_loss,
                    model.adv_G_loss,
                    model.good_fn_loss,
                    model.evil_fn_loss,
                    model.hinge_loss,
                    model.G_train_op], feed)
            writer.add_summary(summary_str, iteration)

            summary_str, D_loss, _ = sess.run([
                    model.pre_d_loss_sum,
                    model.D_loss,
                    model.D_pre_train_op], feed)
            writer.add_summary(summary_str, iteration)

            if iteration != 0 and iteration % opt.losses_log_every == 0:
                print "iteration: ", iteration
                print "D: %.4f, G: %.4f, pre_G_loss: %.4f, adv_G: %.4f, hinge_loss: %.4f" % (
                    D_loss, G_loss, pre_G_loss, adv_G_loss, hinge_loss)
                print '\tGood loss: %.6f, Evil loss: %.6f' % (good_fn_loss, evil_fn_loss)
                loss_file.write('%d, %.4f, %.4f\n' % (iteration, good_fn_loss, evil_fn_loss))


            if iteration != 0 and iteration % opt.save_checkpoint_every == 0:
                checkpoint_path = os.path.join(opt.checkpoint_path, 'checkpoint.ckpt')
                print 'Saving the model in "%s"' % checkpoint_path

                model.saver.save(sess, checkpoint_path, global_step=iteration)

                test_loader = Dataset2(test_data, test_label)

                test_num = test_loader._num_examples
                test_iter_num = (test_num - batch_size) / batch_size
                # test_iter_num = 1
                total_evil_accuracy = 0.0
                total_good_accuracy = 0.0
                # show_samples = []

                # save_samples = []
                # input_samples = []
                fake_samples = [[] for _ in range(test_loader._num_labels)]
                fake_noise = [[] for _ in range(test_loader._num_labels)]

                print '\t\tfake samples shape is ' + str(fake_samples.shape) 

                for _ in range(test_iter_num):

                    # Loading the next batch of test images
                    test_input_data, test_evil_labels, _ = \
                        test_loader.next_batch(batch_size)
                    evil_categorical_labels = np.argmax(test_evil_labels, axis=1)
                    test_good_labels = odd_even_labels(test_evil_labels)
                    feed = {
                        model.source: test_input_data,
                        model.evil_labels: test_evil_labels,
                        model.good_labels: test_good_labels
                    }

                    evil_accuracy, good_accuracy = sess.run(
                        [model.evil_accuracy, model.good_accuracy], feed)
                    # We divide the total accuracy by the number of test iterations.
                    total_good_accuracy += good_accuracy
                    total_evil_accuracy += evil_accuracy
                    # print 'Evil accuracy: %.6f\tGood accuracy: %.6f' % (
                    #     evil_accuracy, good_accuracy)
                    # test_accuracy, test_adv_accuracy = sess.run(
                    #     [model.accuracy, model.adv_accuracy], feed)
                    # test_acc += test_accuracy
                    # test_adv_acc += test_adv_accuracy

                    fake_images, g_x = sess.run(
                        [model.fake_images, model.g_x],
                        {model.source: test_input_data})
                    print '\t\tThe shape of noise is ' + str(g_x.shape)

                    for lbl in range(test_loader._num_labels):
                        if len(fake_samples[lbl]) < 10:
                            idx = np.where(evil_categorical_labels == lbl)[0]
                            if idx.shape[0] >= 10:
                                fake_samples[lbl] = fake_images[idx[:10]]
                                fake_noise[lbl] = g_x[idx[:10]]


                    # for lbl, sample, noise in zip(test_evil_labels, fake_images, fake_noise):
                    #     if len(fake_samples[lbl]) > 10:
                    #         continue
                    #     fake_samples[lbl].append(sample)
                    #     fake_noise[lbl].append(noise)

                    # pdb.set_trace()
                    # print fake_images.shape

                    # Finding those predicted labels that are equal to the target label
                    # idxs = np.where(out_predict_labels == target_label)[0]
                    # save_images(samples[:100], [10, 10], 'CIFAR10/result2/test_' + str(source_idx) + str(target_idx)+  '_.png')
                    # pdb.set_trace()
                    # show_samples.append(samples)
                    # input_samples.append(s_imgs)
                    # save_samples.append(samples)
                    # if opt.is_advGAN:
                    #     save_samples.append(samples[idxs])
                    # else:
                        # We add all samples.
                # show_samples = np.concatenate(show_samples, axis=0)
                # save_samples = np.concatenate(save_samples, axis=0)
                good_accuracy = total_good_accuracy / float(test_iter_num)
                evil_accuracy = total_evil_accuracy / float(test_iter_num)
                print '\tAccuracy diff: %f' % (good_accuracy - evil_accuracy)
                print '\tGood accuracy %f, Evil accuracy %f' % (
                    good_accuracy, evil_accuracy)
                acc_file.write('%d, %.4f, %.4f\n' % (
                    iteration, good_accuracy, evil_accuracy))

                # Resizing the samples to save them later on.
                fake_samples = np.reshape(np.array(fake_samples), [100, -1])
                fake_noise = np.reshape(np.array(fake_noise), [100, -1])

                if (good_accuracy - evil_accuracy) > max_accuracy_diff:
                    max_accuracy_diff = good_accuracy - evil_accuracy
                    # test_accuracy = test_acc / float(test_iter_num)
                    # test_adv_accuracy = test_adv_acc / float(test_iter_num)
                    # if (good_accuracy - evil_accuracy) > max_accuracy_diff:
                    #     max_accuracy_diff = good_accuracy - evil_accuracy
                    # if min_adv_accuracy > test_adv_accuracy:
                    #     min_adv_accuracy = test_adv_accuracy
                    # save_images(fake_images[:100], [10, 10], 'fake.png')
                    # save_images(test_input_data[:100], [10, 10], 'real.png')
                    save_images(fake_samples[:100], [10, 10], 'best_images.png')
                    save_images(fake_noise[:100], [10, 10], 'best_noise.png')
                    save_anything = True
                    # Saving the best yet model.
                    best_model_path = os.path.join(opt.checkpoint_path, 'best.ckpt')
                    print 'Saving the best model yet at "%s"' % best_model_path
                    model.saver.save(sess, best_model_path)

                if save_anything is False:
                    # Nothing is saved. We save a version here.
                    save_images(fake_samples[:100], [10, 10], 'last_images.png')
                    save_images(fake_noise[:100], [10, 10], 'last_noise.png')
                    save_anything = True

            iteration += 1
        acc_file.close()
        loss_file.close()

if __name__ == "__main__":
    train()
