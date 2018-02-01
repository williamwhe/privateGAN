from __future__ import division

import os

from scipy.misc import imsave as scipy_imsave
import numpy as np
import tensorflow as tf

from ops import *
import opts

from utils import merge
from advgan import advGAN
from Dataset2 import Dataset2
from lfw import get_30_people_chunk, balance_dataset
from face_recognizer import FaceRecognizer

def get_output_samples(imgs, lbls, id_gender, num_repr, num_samples_each):
    id_gender_ct = np.argmax(id_gender, axis=1)
    lbls_ct = np.argmax(lbls, axis=1)
    print lbls_ct.shape
    print id_gender_ct
    samples = []
    for i in range(2):
        indices = np.random.choice(np.where(id_gender_ct == i)[0],
                                   num_repr,
                                   replace=False)
        for idx in indices:
            selection = np.random.choice(np.where(lbls_ct == idx)[0],
                                         num_samples_each,
                                         replace=False)
            samples.append(imgs[selection, :])

    samples = np.concatenate(samples)
    return samples

def train():
    opt = opts.parse_opt()
    opt.input_data = "MNIST"

    img_size = (opt.img_dim, opt.img_dim)
    print 'Dimension of images:', img_size
    train_data, train_label, id_gender = \
        get_30_people_chunk(opt.image_path, 1, gender_meta=True, img_size=img_size)
    test_data, test_label = get_30_people_chunk(opt.image_path, 2, img_size=img_size)


    if opt.balance_data:
        ratio = opt.balance_ratio
        print 'Balancing dataset with ratio %f' % ratio
        train_data, train_label = balance_dataset(train_data, train_label)
        test_data, test_label = balance_dataset(test_data, test_label)

    print 'Shape of data:'
    print '\tTraining data: ' + str(train_data.shape)
    print '\tTraining label: ' + str(train_label.shape)
    print '\tMax, Min Train: %.4f, %.4f' % (np.max(train_data), np.min(train_data))
    print '\tTest data: ' + str(test_data.shape)
    print '\tTest label: ' + str(test_label.shape)
    print '\tMax, Min Test: %.4f, %.4f' % (np.max(test_data), np.min(test_data))

    x_dim = train_data.shape[1]
    y_dim = train_label.shape[1]

    opt.input_c_dim = 3
    opt.output_c_dim = 3
    opt.input_dim = x_dim
    opt.label_dim = y_dim
    input_shape = (x_dim, x_dim, opt.input_c_dim)

    batch_size = opt.batch_size
    print 'Batch size: %d' % batch_size

    NUM_REPR = 5
    NUM_SAMPLES_EACH = int(batch_size / NUM_REPR / 2)
    output_samples = get_output_samples(train_data, train_label, id_gender,
                                        NUM_REPR, NUM_SAMPLES_EACH)

    NUM_THREADS = 2
    tf_config = tf.ConfigProto()
    tf_config.intra_op_parallelism_threads = NUM_THREADS
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:

        id_model_path = '%s_%d_id' % (opt.lfw_base_path, x_dim)
        print '\tRetrieving evil model from "%s"' % id_model_path
        evil_model = FaceRecognizer(id_model_path,
                                    train_label.shape[1],
                                    input_shape,
                                    opt.input_c_dim)

        gender_model_path = '%s_%d_gender' % (opt.lfw_base_path, x_dim)
        print '\tRetrieving good model from "%s"' % gender_model_path
        good_model = FaceRecognizer(gender_model_path, 2, input_shape, opt.input_c_dim)
        model = advGAN(good_model, evil_model, opt, sess, mnist=False)

        iteration = 0
        if opt.resnet_gen:
            generator_mode = 'ResNet'
        else:
            generator_mode = 'Regular'
        summary_dir = "logs/g_%d_ld_%d_gl_%d_L2_%.2f_%s/" % (
            opt.G_lambda, opt.ld, opt.good_loss_coeff, opt.L2_lambda, generator_mode)
        if os.path.isdir(summary_dir) is False:
            print 'Creating directory %s for logs.' % summary_dir
            os.mkdir(summary_dir)
        else:
            print 'Removing all files in %s' % (summary_dir + '*')
            os.remove(summary_dir + '*')

        writer = tf.summary.FileWriter(summary_dir, sess.graph)
        loader = Dataset2(train_data, train_label)
        print 'Training data loaded.'

        print 'Maximum iterations: %d' % opt.max_iteration
        while iteration < opt.max_iteration:
            # this function returns (data, label, np.array(target)).
            feed_data, evil_labels, real_data = loader.next_batch(
                batch_size, negative=False)
            good_labels = id_gender[np.argmax(evil_labels, axis=1)]

            feed = {
                model.source: feed_data,
                model.target: real_data,
                model.good_labels: good_labels,
                model.evil_labels: evil_labels
            }

            # Training G once.
            summary_str, G_loss, _ = sess.run(
                [model.total_loss_merge_sum, model.g_loss, model.G_train_op], feed)
            writer.add_summary(summary_str, iteration)

            # Training G twice.
            summary_str, G_loss, gan_loss, hinge_loss, l1_loss, l2_loss, \
                good_fn_loss, evil_fn_loss, adv_loss, total_loss, _ = sess.run([
                    model.total_loss_merge_sum,
                    model.g_loss,
                    model.gan_loss,
                    model.hinge_loss,
                    model.l1_loss,
                    model.l2_loss,
                    model.good_fn_loss,
                    model.evil_fn_loss,
                    model.adv_loss,
                    model.total_loss,
                    model.G_train_op], feed)
            writer.add_summary(summary_str, iteration)

            # Training D.
            summary_str, D_loss, _ = \
                sess.run([model.total_loss_merge_sum, model.d_loss, model.D_pre_train_op], feed)
            writer.add_summary(summary_str, iteration)

            # if iteration != 0 and iteration % opt.losses_log_every == 0:
            if iteration % opt.losses_log_every == 0:
                print "iteration: ", iteration
                print '\tD: %.4f, G: %.4f\n\thinge(%.2f): %.4f, L1(%.2f): %.4f, L2(%.2f): %.4f' % (
                    D_loss, G_loss, opt.H_lambda, hinge_loss,
                    opt.L1_lambda, l1_loss, opt.L2_lambda, l2_loss)
                print '\t\tGAN total loss: %.4f' % gan_loss
                print '\tGood: %.4f, Evil: %.4f' % (good_fn_loss, evil_fn_loss)
                print '\tAdv: %.4f, Total: %.4f' % (adv_loss, total_loss)

                # model.saver.save(sess, checkpoint_path, global_step=iteration)
                test_loader = Dataset2(test_data, test_label)

                test_num = test_loader._num_examples
                test_iter_num = int(np.ceil((test_num - batch_size) / batch_size))
                total_evil_accuracy = 0.0
                total_good_accuracy = 0.0
                fake_samples = [[] for _ in range(test_loader._num_labels)]
                fake_noise = [[] for _ in range(test_loader._num_labels)]

                total_good_confusion = np.zeros((2, 2))
                for _ in range(test_iter_num):

                    # Loading the next batch of test images
                    test_input_data, test_evil_labels, _ = \
                        test_loader.next_batch(batch_size)
                    evil_categorical_labels = np.argmax(test_evil_labels, axis=1)
                    test_good_labels = id_gender[evil_categorical_labels]
                    feed = {
                        model.source: test_input_data,
                        model.evil_labels: test_evil_labels,
                        model.good_labels: test_good_labels
                    }
                    evil_accuracy, good_accuracy, good_confusion = sess.run(
                        [model.evil_accuracy, model.good_accuracy, model.good_confusion], feed)
                    # We divide the total accuracy by the number of test iterations.
                    total_good_accuracy += good_accuracy
                    total_evil_accuracy += evil_accuracy
                    total_good_confusion += good_confusion

                good_accuracy = total_good_accuracy / float(test_iter_num)
                evil_accuracy = total_evil_accuracy / float(test_iter_num)
                print '\tGood Accuracy: %.4f, Evil Accuracy: %.4f' % (
                    good_accuracy, evil_accuracy)
                print '\tAccuracy diff: %f' % (good_accuracy - evil_accuracy)
                print 'Good confusion matrix:'
                print total_good_confusion

                fake_samples, fake_noise = sess.run(
                    [model.fake_images_output, model.fake_noise_output],
                    {model.source: output_samples})

                fakes = merge(fake_samples, [2 * NUM_REPR, NUM_SAMPLES_EACH])
                original = merge(output_samples, [2 * NUM_REPR, NUM_SAMPLES_EACH])
                noise = merge(fake_noise, [2 * NUM_REPR, NUM_SAMPLES_EACH])
                final_image = np.concatenate([fakes, noise, original], axis=1)
                # np.savez_compressed('./samples_%d.npz' % iteration,
                #                     fakes=fakes,
                #                     original=original,
                #                     noise=noise)

                scipy_imsave('snapshot_%d.png' % iteration, final_image)

            iteration += 1

        # We can transform the training and test data given in the beginning here.
        # This is only half the actual data.
        print "We don't make new datasets for now. This is only training tuning."
        # Not making new dataset for now. First, do some successful training.
        # print 'Making new training data ...',
        # loader = Dataset2(train_data, train_label)
        # iter_num = int((loader._num_examples - batch_size) / batch_size)
        # new_train_data = []
        # new_train_label = []
        # for _ in range(iter_num):
        #     batch_data, batch_label, _ = loader.next_batch(batch_size)
        #     new_data = sess.run(model.fake_images_sample, {model.source: batch_data})
        #     new_train_data.append(new_data)
        #     new_train_label.append(batch_label)
        # new_train_data = np.concatenate(new_train_data)
        # new_train_label = np.concatenate(new_train_label)
        # print '[DONE]'
        # print 'Making new test data ...',
        # loader = Dataset2(test_data, test_label)
        # iter_num = int((loader._num_examples - batch_size) / batch_size)
        # new_test_data = []
        # new_test_label = []
        # for _ in range(iter_num):
        #     batch_data, batch_label, _ = loader.next_batch(batch_size)
        #     new_data = sess.run(model.fake_images_sample, {model.source: batch_data})
        #     new_test_data.append(new_data)
        #     new_test_label.append(batch_label)
        # new_test_data = np.concatenate(new_test_data)
        # new_test_label = np.concatenate(new_test_label)
        # print '[DONE]'

        # print 'Training:'
        # print new_train_data.shape
        # print new_train_label.shape
        # print 'Test:'
        # print new_test_data.shape
        # print new_test_label.shape

        # print 'Saving ...',
        # np.savez_compressed(opt.output_path,
        #                     train_data=new_train_data,
        #                     train_label=new_train_label,
        #                     test_data=new_test_data,
        #                     test_label=new_test_label)
        # print '[DONE]'

if __name__ == "__main__":
    train()
