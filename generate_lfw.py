from __future__ import division

import os
import time
import getpass
if getpass.getuser() == 'aria':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from scipy.misc import imsave as scipy_imsave
import numpy as np
import tensorflow as tf

from ops import *
import opts

from utils import merge
from advgan import advGAN
from Dataset2 import Dataset2
from lfw import get_30_people_chunk, balance_dataset, get_people_names, preprocess_images
from face_recognizer import FaceRecognizer
from sklearn.metrics import accuracy_score, confusion_matrix


def get_output_samples(imgs, lbls, id_gender, num_repr, num_samples_each):
    id_gender_ct = np.argmax(id_gender, axis=1)
    lbls_ct = np.argmax(lbls, axis=1)
    sum_lbl = lbls.sum(axis=0)
    print lbls_ct.shape
    print id_gender_ct
    samples = []
    for i in range(2):
        indices = np.where((id_gender_ct == i) & (sum_lbl != 0))[0][:num_repr]
        for idx in indices:
            selection = np.random.choice(np.where((lbls_ct == idx))[0],
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
    names = get_people_names(opt.image_path, 30)

    if opt.balance_data:
        ratio = opt.balance_ratio
        print 'Balancing dataset with ratio %f' % ratio
        train_data, train_label = balance_dataset(train_data, train_label)
        test_data, test_label = balance_dataset(test_data, test_label)

    if opt.balance_gender:
        print train_data.shape, train_label.shape
        print test_data.shape, test_label.shape
        print 'Balancing genders'
        selected_people = []
        for i in range(id_gender.shape[1]):
            indices, = np.where(id_gender[:, i] == 1)
            selected_people.append(np.random.choice(indices, 5, replace=False))
        selected_people = np.concatenate(selected_people)

        print 'Selected people are:'
        print np.array(names)[selected_people]

        selected_imgs = train_label[:, selected_people].sum(axis=1) != 0
        train_data = train_data[selected_imgs, :]
        train_label = train_label[selected_imgs, :]

        selected_imgs = test_label[:, selected_people].sum(axis=1) != 0
        test_data = test_data[selected_imgs, :]
        test_label = test_label[selected_imgs, :]

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

    iteration_time = []
    with tf.Session(config=tf_config) as sess:

        id_model_path = '%s_%d_id_0' % (opt.lfw_base_path, x_dim)
        print '\tRetrieving evil model from "%s"' % id_model_path
        evil_model = FaceRecognizer(id_model_path,
                                    train_label.shape[1],
                                    input_shape,
                                    opt.input_c_dim)

        gender_model_path = '%s_%d_gender_0' % (opt.lfw_base_path, x_dim)
        print '\tRetrieving good model from "%s"' % gender_model_path
        good_model = FaceRecognizer(gender_model_path, 2, input_shape, opt.input_c_dim)
        model = advGAN(good_model, evil_model, opt, sess, mnist=False)

        iteration = 0
        if opt.resnet_gen:
            generator_mode = 'ResNet'
        else:
            generator_mode = 'Regular'
        summary_dir = "logs/LFW/g_%d_ld_%d_gl_%d_L2_%.2f_lr_%.4f_%s/" % (
            opt.G_lambda, opt.ld, opt.good_loss_coeff,
            opt.L2_lambda, opt.learning_rate, generator_mode)
        if os.path.isdir(summary_dir) is False:
            print 'Creating directory %s for logs.' % summary_dir
            os.mkdir(summary_dir)
        # else:
        #     print 'Removing all files in %s' % (summary_dir + '*')
        #     shutil.rmtree(summary_dir)

        writer = tf.summary.FileWriter(summary_dir, sess.graph)
        loader = Dataset2(train_data, train_label)
        print 'Training data loaded.'

        print 'Maximum iterations: %d' % opt.max_iteration
        max_acc_diff = -1.0
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

            if iteration % opt.losses_log_every == 0:
                print "iteration: ", iteration
                print '\tD: %.4f, G: %.4f\n\thinge(%.2f): %.4f, L1(%.2f): %.4f, L2(%.2f): %.4f' % (
                    D_loss, G_loss, opt.H_lambda, hinge_loss,
                    opt.L1_lambda, l1_loss, opt.L2_lambda, l2_loss)
                print '\t\tGAN total loss: %.4f' % gan_loss
                print '\tGood: %.4f, Evil: %.4f' % (good_fn_loss, evil_fn_loss)
                print '\tAdv: %.4f, Total: %.4f' % (adv_loss, total_loss)

                new_test_data = []
                new_pred_data = []
                head = 0
                last_batch = False
                while head < test_data.shape[0]:
                    if head + batch_size <= test_data.shape[0]:
                        tail = head + batch_size
                    else:
                        tail = test_data.shape[0]
                        head = test_data.shape[0] - batch_size
                        last_batch = True
                    cur_data, pred_data = sess.run(
                        [model.fake_images_output, model.prediction_ready],
                        {model.source: test_data[head:tail, :]})

                    if last_batch:
                        new_test_data.append(
                            cur_data[-(test_data.shape[0] % batch_size):, :])
                        new_pred_data.append(
                            pred_data[-(test_data.shape[0] % batch_size):, :])
                    else:
                        new_test_data.append(cur_data)
                        new_pred_data.append(pred_data)
                    head += batch_size
                new_test_data = np.concatenate(new_test_data)
                new_pred_data = np.concatenate(new_pred_data)

                good_pred = np.argmax(model.good_model.model.predict(new_pred_data), axis=1)
                evil_pred = np.argmax(model.evil_model.model.predict(new_pred_data), axis=1)
                evil_true = np.argmax(test_label, axis=1)
                good_true = np.argmax(id_gender[evil_true, :], axis=1)

                good_accuracy = accuracy_score(good_true, good_pred)
                evil_accuracy = accuracy_score(evil_true, evil_pred)
                total_good_confusion = confusion_matrix(good_true, good_pred)
                total_evil_confusion = confusion_matrix(evil_true, evil_pred,
                                                        labels=range(opt.evil_label_num))

                print '\tGood Accuracy: %.4f, Evil Accuracy: %.4f' % (
                    good_accuracy, evil_accuracy)
                print '\tAccuracy diff: %f' % (good_accuracy - evil_accuracy)
                print 'Good confusion matrix:'
                print total_good_confusion
                evil_misclass = total_evil_confusion.sum(axis=0) - np.diag(total_evil_confusion)
                evil_idxs = np.argsort(-evil_misclass)
                print 'Top 3 Misclassifications:'
                print np.array(names)[evil_idxs][:3]
                print evil_misclass[evil_idxs][:3]
                evil_tp = np.diag(total_evil_confusion)
                evil_idxs = np.argsort(-evil_tp)
                print 'Top 3 True classifications:'
                print np.array(names)[evil_idxs][:3]
                print evil_tp[evil_idxs][:3]

                # print 'Selected people are:'
                # print names[evil_idxs].tolist()
                # print evil_tp
                # print total_evil_confusion
                # print evil_idxs

                fake_samples, fake_noise = sess.run(
                    [model.fake_images_output, model.fake_noise_output],
                    {model.source: output_samples})

                fakes = merge(fake_samples, [2 * NUM_REPR, NUM_SAMPLES_EACH])
                original = merge(output_samples, [2 * NUM_REPR, NUM_SAMPLES_EACH])
                noise = merge(fake_noise, [2 * NUM_REPR, NUM_SAMPLES_EACH])
                final_image = np.concatenate([fakes, noise, original], axis=1)

                scipy_imsave('snapshot_%d.png' % iteration, final_image)

                if (good_accuracy - evil_accuracy) > max(0.5, max_acc_diff):
                    print '\tSaving new training data at accuracy diff: %.4f' % (
                        good_accuracy - evil_accuracy),
                    max_acc_diff = good_accuracy - evil_accuracy

                    # other_good = FaceRecognizer('%s_%d_gender_0' % (opt.lfw_base_path, x_dim),
                    #                             2, input_shape, opt.input_c_dim)

                    # other_pred = np.argmax(other_good.model.predict(new_pred_data), axis=1)
                    # print 'Other Good accuracy: %.4f' % accuracy_score(good_true, other_pred)

                    # other_pred = np.argmax(other_good.model.predict(
                    #     preprocess_images(new_test_data * 255.0)), axis=1)
                    # print '\tTest data processeced accuracy: %.4f' % \
                    #     accuracy_score(good_true, other_pred)

                    # other_evil = FaceRecognizer('%s_%d_id_0' % (opt.lfw_base_path, x_dim),
                    #                             34, input_shape, opt.input_c_dim)
                    # other_pred = np.argmax(other_evil.model.predict(new_pred_data), axis=1)
                    # print 'Other Evil accuracy: %.4f' % accuracy_score(evil_true, other_pred)
                    # other_pred = np.argmax(other_evil.model.predict(
                    #     preprocess_images(new_test_data * 255.0)), axis=1)
                    # print '\tTest data processeced accuracy: %.4f' % \
                    #     accuracy_score(evil_true, other_pred)

                    new_train_data = []
                    head = 0
                    last_batch = False
                    while head < train_data.shape[0]:
                        if head + batch_size <= train_data.shape[0]:
                            tail = head + batch_size
                        else:
                            tail = train_data.shape[0]
                            head = train_data.shape[0] - batch_size
                            last_batch = True
                        cur_data = sess.run(
                            model.fake_images_output,
                            {model.source: train_data[head:tail, :]})

                        if last_batch:
                            new_train_data.append(
                                cur_data[-(train_data.shape[0] % batch_size):, :])
                        else:
                            new_train_data.append(cur_data)
                        head += batch_size
                    new_train_data = np.concatenate(new_train_data)

                    np.savez_compressed(opt.output_path,
                                        train_data=new_train_data,
                                        org_train_data=train_data,
                                        train_label=train_label,
                                        test_data=new_test_data,
                                        org_test_data=test_data,
                                        test_label=test_label,
                                        id_gender=id_gender)
                    print '\t[DONE]'

            iteration += 1

if __name__ == "__main__":
    train()
