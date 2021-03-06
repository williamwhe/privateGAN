import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from ops import *
import scipy
import opts
# from utils import plot
from utils import merge
from advgan import advGAN
from Dataset2 import Dataset2, odd_even_labels, output_sample
# from sklearn import model_selection
from setup_mnist import MNIST, MNISTModel, MNISTModel2, MNISTModel3, OddEvenMNIST
from setup_cifar import CIFARModel, CIFARModel2, CIFARModel3
from sklearn.metrics import accuracy_score, confusion_matrix

tag_num = 0

def train():
    flatten_flag = True  # flatten output of G or not?
    opt = opts.parse_opt()
    opt.input_data = "MNIST"
    # mapping [0,1] -> [-1,1]
    # load data
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # train_data = mnist.train.images * 2.0 - 1.0
    # train_label = mnist.train.labels

    # test_data = mnist.test.images * 2.0 - 1.0
    # test_label = mnist.test.labels

    loaded = np.load('MNIST_data/B.npz')
    train_data, train_label, test_data, test_label = \
        loaded['train_data'], loaded['train_label'], \
        loaded['test_data'], loaded['test_label']

    # We create the label clues here.
    if opt.cgan_gen is True:
        label_clue = np.zeros((
            train_label.shape[1], opt.img_dim, opt.img_dim, train_label.shape[1]))
        for lbl in range(train_label.shape[1]):
            label_clue[lbl, :, :, lbl] = 1

    if opt.cgan_gen:
        output_samples, output_labels = output_sample(test_data, test_label, True)
    else:
        output_samples = output_sample(test_data, test_label)
    print output_samples.shape

    print 'Shape of data:'
    print '\tTraining data: ' + str(train_data.shape)
    print '\tTraining label: ' + str(train_label.shape)
    print '\tTest data: ' + str(test_data.shape)
    print '\tTest label: ' + str(test_label.shape)

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
        global_step = 0

        print '\tRetrieving evil model from "%s"' % opt.evil_model_path
        evil_model = MNISTModel(opt.evil_model_path)
        print '\tRetrieving good model from "%s"' % opt.good_model_path
        good_model = OddEvenMNIST(opt.good_model_path)
        # model = advGAN(whitebox_model, model_store, opt, sess)
        model = advGAN(good_model, evil_model, opt, sess)

        min_adv_accuracy = 10e10
        max_accuracy_diff = -np.inf

        # summary_dir = "logs/MNIST/g_%d_ld_%d_gl_%d_L2_%.2f_dn_%d" % (
        #     opt.G_lambda, opt.ld, opt.good_loss_coeff,
        #     opt.L2_lambda, opt.d_train_num)

        summary_dir = "logs/MNIST/dn_%d_gn_%d" % (
            opt.d_train_num, opt.g_train_num)

        duplicate_num = 0
        while os.path.isdir(summary_dir + '_' + str(duplicate_num) + '/'):
            duplicate_num += 1
        summary_dir += '_' + str(duplicate_num) + '/'
        print 'Creating directory %s for logs.' % summary_dir
        os.mkdir(summary_dir)

        writer = tf.summary.FileWriter(summary_dir, sess.graph)
        loader = Dataset2(train_data, train_label)
        print 'Training data loaded.'

        best_evil_accuracy = -1.0
        best_res_epoch = -1
        best_res = None
        for epoch_num in range(opt.max_epoch):
            print 'Epoch %d' % epoch_num

            # Randomly shuffle the data.
            random_indices = np.arange(train_data.shape[0])
            np.random.shuffle(random_indices)
            train_data = train_data[random_indices, :]
            train_label = train_label[random_indices, :]

            real_buckets = []
            for lbl in range(train_label.shape[1]):
                real_buckets.append(np.where(train_label[:, lbl] == 1)[0])

            # Mini-batch Gradient Descent.
            batch_no = 0
            while (batch_no * batch_size) < train_data.shape[0]:
                head = batch_no * batch_size
                if head + batch_size <= train_data.shape[0]:
                    tail = head + batch_size
                else:
                    tail = train_data.shape[0]
                    head = train_data.shape[0] - batch_size

                feed_data = train_data[head:tail, :]
                evil_labels = train_label[head:tail, :]
                good_labels = odd_even_labels(evil_labels)

                # Finding randomly sampled real data.
                real_data = np.zeros_like(feed_data)
                # Indices of training batch with specific label.
                # label_indices[i] = indices of feed data, that have evil_label[i] == 1.
                label_indices = [np.where(evil_labels[:, lbl] == 1)[0] \
                    for lbl in range(evil_labels.shape[1])]

                for lbl in range(evil_labels.shape[1]):
                    # We take a random sample of size |label_indices[lbl]|
                    # from the real bucket of `lbl`.
                    selected_real_data = np.random.choice(
                        real_buckets[lbl], label_indices[lbl].shape[0])

                    # We put this random sample in the same index of their
                    # corresponding batch training data.
                    real_data[label_indices[lbl], :] = train_data[selected_real_data, :]

                feed = {
                    model.source: feed_data,
                    model.target: real_data,
                    model.good_labels: good_labels,
                    model.evil_labels: evil_labels
                }

                # Train G.
                for _ in range(opt.g_train_num):
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
                    writer.add_summary(summary_str, global_step)

                # Train D.
                for _ in range(opt.d_train_num):
                    summary_str, D_loss, _ = sess.run(
                        [model.total_loss_merge_sum, model.d_loss, model.D_pre_train_op], feed)
                    writer.add_summary(summary_str, global_step)

                global_step += 1
                batch_no += 1

            # Validation after each trainig epoch.
            print '\tD: %.4f, G: %.4f\n\thinge(%.1f): %.4f, L1(%.1f): %.4f, L2(%.1f): %.4f' % (
                D_loss, G_loss, opt.H_lambda, hinge_loss,
                opt.L1_lambda, l1_loss, opt.L2_lambda, l2_loss)
            print '\t\tGAN total loss: %.4f' % gan_loss
            print '\tGood: %.4f, Evil: %.4f' % (good_fn_loss, evil_fn_loss)
            print '\tAdv: %.4f, Total: %.4f' % (adv_loss, total_loss)

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
                if opt.cgan_gen:
                    cur_data = sess.run(
                        model.fake_images_sample,
                        {model.evil_labels: test_label[head:tail, :]})
                else:
                    cur_data = sess.run(
                        model.fake_images_sample,
                        {model.source: test_data[head:tail, :]})

                if last_batch:
                    new_pred_data.append(
                        cur_data[-(test_data.shape[0] % batch_size):, :])
                else:
                    new_pred_data.append(cur_data)
                head += batch_size
            new_pred_data = np.concatenate(new_pred_data)

            good_pred = np.argmax(model.good_model.model.predict(new_pred_data), axis=1)
            evil_pred = np.argmax(model.evil_model.model.predict(new_pred_data), axis=1)
            evil_true = np.argmax(test_label, axis=1)
            good_true = np.argmax(odd_even_labels(test_label), axis=1)

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
            print 'Evil confusion matrix:'
            print total_evil_confusion

            # Creating snapshots to save.
            if opt.cgan_gen:
                fake_samples = sess.run(
                    model.fake_images_sample,
                    {model.evil_labels: output_labels})
            else:
                fake_samples, fake_noise = sess.run(
                    [model.fake_images_sample, model.sample_noise],
                    {model.source: output_samples})
            max_accuracy_diff = good_accuracy - evil_accuracy

            fakes = merge(fake_samples[:100, :], [10, 10])
            separator = np.ones((280, 2))
            original = merge(output_samples[:100].reshape(-1, 28, 28, 1), [10, 10])

            if opt.cgan_gen:
                scipy.misc.imsave(
                    'snapshot_%d.png' % epoch_num,
                    np.concatenate([fakes, separator, original], axis=1))
            else:
                noise = merge(fake_noise[:100], [10, 10])
                scipy.misc.imsave(
                    'snapshot_%d.png' % epoch_num,
                    np.concatenate([fakes, noise, original], axis=1))

            # Only for the purpose of finding best D and G training times.
            if evil_accuracy > best_evil_accuracy:
                best_evil_accuracy = evil_accuracy
                best_res_epoch = epoch_num
                if opt.cgan_gen:
                    best_res = np.concatenate([fakes, separator, original], axis=1)
                else:
                    best_res = np.concatenate([fakes, noise, original], axis=1)

        best_image_path = 'best_dn_%d_gn_%d_%d_epoch_%d.png' % \
            (opt.d_train_num, opt.g_train_num, duplicate_num, best_res_epoch)
        scipy.misc.imsave(best_image_path, best_res)

        # print 'Maximum iterations: %d' % opt.max_iteration
        # while iteration < opt.max_iteration:
        #     # this function returns (data, label, np.array(target)).
        #     # data = loader.next_batch(batch_size, negative=False)
        #     feed_data, evil_labels, real_data = loader.next_batch(
        #         batch_size, negative=False)
        #     good_labels = odd_even_labels(evil_labels)

        #     feed = {
        #         model.source: feed_data,
        #         model.target: real_data,
        #         model.good_labels: good_labels,
        #         model.evil_labels: evil_labels
        #     }

        #     # if opt.cgan_gen:
        #     #     feed[model.label_clue] = label_clue[evil_labels.argmax(axis=1)]

        #     # Training G once.
        #     # summary_str, G_loss, _ = sess.run(
        #     #     [model.total_loss_merge_sum, model.g_loss, model.G_train_op], feed)
        #     # writer.add_summary(summary_str, iteration)

        #     # Training G twice.
        #     summary_str, G_loss, gan_loss, hinge_loss, l1_loss, l2_loss, \
        #         good_fn_loss, evil_fn_loss, adv_loss, total_loss, _ = sess.run([
        #             model.total_loss_merge_sum,
        #             model.g_loss,
        #             model.gan_loss,
        #             model.hinge_loss,
        #             model.l1_loss,
        #             model.l2_loss,
        #             model.good_fn_loss,
        #             model.evil_fn_loss,
        #             model.adv_loss,
        #             model.total_loss,
        #             model.G_train_op], feed)
        #     writer.add_summary(summary_str, iteration)

        #     # Training D.
        #     for _ in range(opt.d_train_num):
        #         summary_str, D_loss, _ = sess.run(
        #             [model.total_loss_merge_sum, model.d_loss, model.D_pre_train_op], feed)
        #         writer.add_summary(summary_str, iteration)

        #     if iteration % opt.losses_log_every == 0:

        #     # if iteration != 0 and iteration % opt.save_checkpoint_every == 0:
        #         # checkpoint_path = os.path.join(opt.checkpoint_path, 'checkpoint.ckpt')
        #         # print 'Saving the model in "%s"' % checkpoint_path

        #         # model.saver.save(sess, checkpoint_path, global_step=iteration)
        #         # test_loader = Dataset2(test_data, test_label)

        #         # test_num = test_loader._num_examples
        #         # test_iter_num = (test_num - batch_size) / batch_size

        #         # total_evil_accuracy = 0.0
        #         # total_good_accuracy = 0.0

        #         # fake_samples = [[] for _ in range(test_loader._num_labels)]
        #         # fake_noise = [[] for _ in range(test_loader._num_labels)]
        #         # original_samples = [[] for _ in range(test_loader._num_labels)]

        #         # for _ in range(test_iter_num):

        #         #     # Loading the next batch of test images
        #         #     test_input_data, test_evil_labels, _ = \
        #         #         test_loader.next_batch(batch_size)
        #         #     evil_categorical_labels = np.argmax(test_evil_labels, axis=1)
        #         #     test_good_labels = odd_even_labels(test_evil_labels)
        #         #     feed = {
        #         #         model.source: test_input_data,
        #         #         model.evil_labels: test_evil_labels,
        #         #         model.good_labels: test_good_labels
        #         #     }

        #         #     # if opt.cgan_gen:
        #         #     #     feed[model.label_clue] = label_clue[test_evil_labels.argmax(axis=1)]

        #         #     evil_accuracy, good_accuracy = sess.run(
        #         #         [model.evil_accuracy, model.good_accuracy], feed)
        #         #     # We divide the total accuracy by the number of test iterations.
        #         #     total_good_accuracy += good_accuracy
        #         #     total_evil_accuracy += evil_accuracy
        #         #     # print 'Evil accuracy: %.6f\tGood accuracy: %.6f' % (
        #         #     #     evil_accuracy, good_accuracy)
        #         #     # test_accuracy, test_adv_accuracy = sess.run(
        #         #     #     [model.accuracy, model.adv_accuracy], feed)
        #         #     # test_acc += test_accuracy
        #         #     # test_adv_acc += test_adv_accuracy

        #         #     # fake_images, g_x = sess.run(
        #         #     #     [model.fake_images_sample, model.sample_noise],
        #         #     #     {model.source: test_input_data})

        #         #     # for lbl in range(test_loader._num_labels):
        #         #     #     if len(fake_samples[lbl]) < 10:
        #         #     #         idx = np.where(evil_categorical_labels == lbl)[0]
        #         #     #         if idx.shape[0] >= 10:
        #         #     #             fake_samples[lbl] = fake_images[idx[:10]]
        #         #     #             fake_noise[lbl] = g_x[idx[:10]]
        #         #     #             original_samples[lbl] = test_input_data[idx[:10]]


        #         #     # for lbl, sample, noise in zip(test_evil_labels, fake_images, fake_noise):
        #         #     #     if len(fake_samples[lbl]) > 10:
        #         #     #         continue
        #         #     #     fake_samples[lbl].append(sample)
        #         #     #     fake_noise[lbl].append(noise)

        #         #     # pdb.set_trace()
        #         #     # print fake_images.shape

        #         #     # Finding those predicted labels that are equal to the target label
        #         #     # idxs = np.where(out_predict_labels == target_label)[0]
        #         #     # save_images(samples[:100], [10, 10], 'CIFAR10/result2/test_' + str(source_idx) + str(target_idx)+  '_.png')
        #         #     # pdb.set_trace()
        #         #     # show_samples.append(samples)
        #         #     # input_samples.append(s_imgs)
        #         #     # save_samples.append(samples)
        #         #     # if opt.is_advGAN:
        #         #     #     save_samples.append(samples[idxs])
        #         #     # else:
        #         #         # We add all samples.
        #         # # show_samples = np.concatenate(show_samples, axis=0)
        #         # # save_samples = np.concatenate(save_samples, axis=0)
        #         # good_accuracy = total_good_accuracy / float(test_iter_num)
        #         # evil_accuracy = total_evil_accuracy / float(test_iter_num)
        #         # print '\tAccuracy diff: %f' % (good_accuracy - evil_accuracy)
        #         # print '\tGood accuracy %f, Evil accuracy %f' % (
        #         #     good_accuracy, evil_accuracy)

        #         # Resizing the samples to save them later on.
        #         # fake_samples = np.reshape(np.array(fake_samples), [100, -1])
        #         # original_samples = np.reshape(np.array(original_samples), [100, -1])
        #         # fake_noise = np.reshape(np.array(fake_noise), [100, -1])

        #         # if (good_accuracy - evil_accuracy) > max_accuracy_diff:
        #         # test_accuracy = test_acc / float(test_iter_num)
        #         # test_adv_accuracy = test_adv_acc / float(test_iter_num)
        #         # if (good_accuracy - evil_accuracy) > max_accuracy_diff:
        #         #     max_accuracy_diff = good_accuracy - evil_accuracy
        #         # if min_adv_accuracy > test_adv_accuracy:
        #         #     min_adv_accuracy = test_adv_accuracy
        #         # save_images(fake_images[:100], [10, 10], 'fake.png')
        #         # save_images(test_input_data[:100], [10, 10], 'real.png')
        #         # all_idx = np.arange(100)
        #         # odds = np.where((all_idx / 10) % 2 == 1)[0]
        #         # evens = np.where((all_idx / 10) % 2 == 0)[0]
        #         # order = np.concatenate((odds, evens))
        #         # save_images(fake_samples[order], [10, 10], 'best_images.png')
        #         # save_images(fake_noise[order], [10, 10], 'best_noise.png')
        #         # save_images(original_samples[order], [10, 10], 'best_original.png')

        #         # save_anything = True
        #         # Saving the best yet model.
        #         # best_model_path = os.path.join(opt.checkpoint_path, 'best.ckpt')
        #         # print 'Saving the best model yet at "%s"' % best_model_path
        #         # model.saver.save(sess, best_model_path)

        #         # if save_anything is False:
        #         #     # Nothing is saved. We save a version here.
        #         #     save_images(fake_samples[:100], [10, 10], 'last_images.png')
        #         #     save_images(fake_noise[:100], [10, 10], 'last_noise.png')
        #         #     save_anything = True

        #     iteration += 1

        # We can transform the training and test data given in the beginning here.
        # This is only half the actual data.
        if opt.save_data:
            # if opt.cgan_gen:
            raise NotImplementedError('Saving data for CGAN_GEN is not yet implemented.')
            # print 'Making new training data ...',
            # loader = Dataset2(train_data, train_label)
            # iter_num = (loader._num_examples - batch_size) / batch_size
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
            # iter_num = (loader._num_examples - batch_size) / batch_size
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
