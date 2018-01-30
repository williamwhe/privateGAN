import numpy as np
import tensorflow as tf
from ops import *
import os
import time
import scipy.io  as sio
import random

# pylint: disable = C0103, W0201, C0111

class advGAN():
    """
    GAN for generating malware.
    good_model: Safe classifier. we try to hold accuracy for this.
    evil_model: adversarial classifier. we try to reduce accuracy for this.
    restore: restore file path for whitebox model
    opts: advGAN parameters. See opts.py.
    sess: TensorFlow Session() object.
    """
    def __init__(self, good_model, evil_model, opts, sess, mnist=True):
        """
        :param D: the discriminator object
        :param params: the dict used to train the generative neural networks
        """
        self.good_model = good_model
        self.evil_model = evil_model
        self.opts = opts
        self.sess = sess
        self.mnist = mnist  # Boolean showing whether or not this is for MNIST.
        self.resnet_gen = opts.resnet_gen
        # self.model_restore = restore

        ####  test flags  ####
        # flag = 0 # concate flag

        self.gf_dim = self.opts.gf_dim
        self.df_dim = self.opts.df_dim

        self.output_c_dim = self.opts.output_c_dim
        self.batch_size = self.opts.batch_size

        self._build_model()
        # net_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope="evagan")
        self.saver = tf.train.Saver(max_to_keep=1000)
        self.sess.run(tf.initialize_all_variables())

        self.good_model.model.load_weights(self.good_model.model_path)
        self.evil_model.model.load_weights(self.evil_model.model_path)

    # def reset(self):
    #     """
    #     Resets the training procedure.
    #     """
    #     # net_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope="evagan")
    #     self.saver = tf.train.Saver(max_to_keep=1)
    #     self.sess.run(tf.initialize_all_variables())
    #     self.model.model.load_weights(self.model_restore)

    # def load(self, checkpoint_path):
    #     """
    #     Lodas from a given checkpoint.
    #     """
    #     self.saver.restore(self.sess, checkpoint_path)
    #     self.model.model.load_weights(self.model_restore)

    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        """
        Initializes a Variable with a truncated normal distribution with a bias.
        """
        return tf.Variable(
            tf.truncated_normal(
                [dim_in, dim_out],
                stddev=stddev/tf.sqrt(float(dim_in)/2.)
            ),
            name=name
        )

    def init_bias(self, dim_out, name=None):
        """
        Initializes the bias with ALL ZEROs.
        """
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def _create_variable(self):
        """Creates variable tensors"""
        ###batch normalization

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn5 = batch_norm(name='d_bn5')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d0 = batch_norm(name='g_bn_d0')
        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        print 'Learning rate is %f' % self.opts.learning_rate
        self.lr = tf.Variable(self.opts.learning_rate, trainable=False, name="learning_rate")


    def _create_placeholder(self):
        """Creates placeholders"""
        input_c_dim = self.opts.input_c_dim
        input_dim = self.opts.input_dim
        label_dim = self.opts.label_dim
        good_label_num = self.opts.good_label_num
        evil_label_num = self.opts.evil_label_num
        img_dim = self.opts.img_dim

        print 'Creating placeholders:'
        print '\timg_dim:', img_dim
        print '\tinput_c_dim', input_c_dim

        if self.mnist:
            # the source image which we want to attack.
            self.source = tf.placeholder(
                tf.float32, [None, input_dim], name="source_image")
            # resize to img_dim x img_dim x img_color_dim (1 in Grayscale images)
            self.images = tf.reshape(
                self.source, [-1, img_dim, img_dim, input_c_dim])

            # the target images, shuffled with the same labels.
            # this is to learn the distribution, not simply images.
            self.target = tf.placeholder(
                tf.float32, [None, input_dim], name='target_image')
            self.real_images = tf.reshape(
                self.target, [-1, img_dim, img_dim, input_c_dim])
        else:
            # No need for resizes. The input is n-dimensional.
            self.source = tf.placeholder(
                tf.float32, [None, img_dim, img_dim, input_c_dim])
            # Scale (0, 1) to (-1, 1)
            self.images = (self.source) * 2.0 - 1
            self.target = tf.placeholder(
                tf.float32, [None, img_dim, img_dim, input_c_dim])
            # Scale (0, 1) to (-1, 1)
            self.real_images = (self.target) * 2.0 - 1


        # labels for the evil classifier.
        self.evil_labels = tf.placeholder(
            tf.float32, [None, evil_label_num], name="evil_label")
        # labels for the good classifier.
        self.good_labels = tf.placeholder(
            tf.float32, [None, good_label_num], name="good_label")

        # used for showing  accuracy.
        self.acc = tf.placeholder(tf.float32, name="accuracy")
        self.acc_sum = tf.summary.scalar("accuracy", self.acc)

        self.adv_acc = tf.placeholder(tf.float32, name="adv_accuracy")
        self.adv_acc_sum = tf.summary.scalar("adv_accuracy", self.adv_acc)

        self.mag = tf.placeholder(tf.float32, name="magnitude")
        self.magnitude_sum = tf.summary.scalar("magnitude", self.mag)

        self.dis = tf.placeholder(tf.float32, name="disortion")
        self.dis_sum = tf.summary.scalar("disortion", self.dis)

        self.metric_sum = tf.summary.merge(
            [self.acc_sum, self.adv_acc_sum, self.magnitude_sum, self.dis_sum])

    def _GAN_model(self, images, fake_images, real_images, g_x):
        cgan_flag = self.opts.cgan_flag
        patch_flag = self.opts.patch_flag
        hinge_flag = self.opts.hinge_flag

        if patch_flag:
        # This is by default True.
            if cgan_flag:
            # This is by default True.
                D_fake_loss, D_fake_logit = self.patch_discriminator(
                    tf.concat([fake_images, images], axis=3))
                D_real_loss, D_real_logit = self.patch_discriminator(
                    tf.concat([real_images, images], axis=3), reuse=True)
            else:
            # This is by default False.
                D_fake_loss, D_fake_logit = \
                    self.patch_discriminator(fake_images)
                D_real_loss, D_real_logit = \
                    self.patch_discriminator(real_images, reuse=True)
        else:
        # This is by default False.
            if cgan_flag:
            # This is by default True.
                D_fake_loss, D_fake_logit = self.discriminator(
                    tf.concat([fake_images, images], axis=3))
                D_real_loss, D_real_logit = self.discriminator(
                    tf.concat([real_images, images], axis=3), reuse=True)
            else:
            # This is by default False.
                D_fake_loss, D_fake_logit = self.discriminator(fake_images)
                D_real_loss, D_real_logit = \
                    self.discriminator(real_images, reuse=True)

        # Discriminator tries to assign 0 to fake data.
        D_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_fake_logit,
                labels=tf.zeros_like(D_fake_logit)))
        # Discriminator tries to assign 1 to real data.
        D_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_real_logit,
                labels=0.9*tf.ones_like(D_real_logit)))
        # D tries to minimize the overall loss.
        D_loss = D_real_loss + D_fake_loss

        # Generator tries to fool Discriminator to assign 1 to fake data.
        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_fake_logit,
                labels=0.9*tf.ones_like(D_fake_logit)))

        L2_norm = tf.reduce_mean(
            tf.reduce_sum(tf.square(fake_images - images), [1, 2, 3]))
        L1_norm = tf.reduce_mean(
            tf.reduce_sum(tf.abs(fake_images - images), [1, 2, 3]))

        # soft bound for l-infinity.
        hinge_loss = tf.maximum(
            (self.L2_dist(fake_images - images) - self.opts.bound), 0)

        return G_loss, D_loss, L1_norm, L2_norm, hinge_loss

    def L2_dist(self, images):
        return tf.reduce_mean(tf.reduce_sum(tf.square(images), [1, 2, 3]))

    def L_infinity(self, image):
        return tf.reduce_max(tf.abs(image))

    def reranking(self, predict_labels, target_labels):

        real = tf.reduce_sum(predict_labels * target_labels, 1)
        other = tf.reduce_max(
            (1 - target_labels) * predict_labels - target_labels * 1000, 1)

        if self.opts.targeted == 1:
            # targeted losss:
            loss = tf.maximum(0.0, other - real + self.opts.confidence)
        else:
            # untargeted loss:
            loss = tf.maximum(0.0, real - other + self.opts.confidence)
        return loss

    def _adversarial_g_loss(self, adv_fake_predict_labels, labels):
        adv_G_loss = tf.reduce_mean(self.reranking(adv_fake_predict_labels, labels))
        return adv_G_loss

    # Average number of correct predictions (Accuracy).
    def _metric(self, labels, predict_labels):
        y_true_cls = tf.argmax(labels, dimension=1)
        y_pred_cls = tf.argmax(predict_labels, dimension=1)
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def _build_model(self):
        """
        builds the advGAN model.
        """
        ld = self.opts.ld
        L1_lambda = self.opts.L1_lambda
        L2_lambda = self.opts.L2_lambda
        G_lambda = self.opts.G_lambda
        hinge_lambda = self.opts.H_lambda

        with tf.variable_scope('evagan'):
            self._create_variable()
            self._create_placeholder()

            # Creating a Generator instance that creates fake images.
            if self.resnet_gen:
                self.fake_images, self.g_x = self.generator_resnet(self.images)
            else:
                self.fake_images, self.g_x = self.generator(self.images)

            # We sample an image
            if self.resnet_gen:
                self.fake_images_sample, self.sample_noise = self.sampler_resnet(self.images)
            else:
                # We use regular samplers.
                self.fake_images_sample, self.sample_noise = self.sampler(self.images)

            # *** FOR LFW ***
            # We scale the fake images and the noise from (-1, 1) to (0, 1) for output.
            self.fake_images_output = (self.fake_images_sample + 1) / 2.0
            self.fake_noise_output = (self.sample_noise + 1) / 2.0

            # self.fake_images_sample_flatten = \
            #     tf.reshape(self.fake_images_sample, [-1, 28 * 28])
            # self.fake_images_correct = \
            #     tf.reshape(self.fake_images, [-1, 28 * 28]) * 0.5 + 0.5

            # self.fake_images_sample_sum = \
            #     tf.summary.image("fake_images", self.fake_images_sample)

            # Creating a GAN model.
            G_loss, D_loss, L1_norm, L2_norm, hinge_loss = self._GAN_model(
                self.images, self.fake_images, self.real_images, self.g_x)
            # G loss and D loss.
            self.l1_loss = L1_norm
            self.l2_loss = L2_norm
            self.g_loss = G_loss
            self.hinge_loss = hinge_loss
            self.gan_loss = G_lambda * G_loss + L1_lambda * L1_norm + \
                L2_lambda * L2_norm + hinge_lambda * hinge_loss

            # test with D_loss
            # self.G_loss = G_loss + D_loss + hinge_lambda * hinge_loss

            self.d_loss = D_loss

            # Adding the values to Summary.
            self.g_loss_sum = tf.summary.scalar("G loss", self.g_loss)
            self.gan_loss_sum = tf.summary.scalar("GAN loss", self.gan_loss)
            self.d_loss_sum = tf.summary.scalar("D loss", self.d_loss)
            self.l1_loss_sum = tf.summary.scalar("l1_loss", self.l1_loss)
            self.l2_loss_sum = tf.summary.scalar("l2_loss", self.l2_loss)
            self.hinge_loss_sum = tf.summary.scalar("hinge_loss", self.hinge_loss)

            # Preparing data for LFW predictions
            self.scaled_up = (self.fake_images + 1) * 127.5
            self.reversed_channels = self.scaled_up[..., ::-1]
            self.prediction_ready = self.reversed_channels - [93.5940, 104.7624, 129.1863]

            # Two competing good/evil classifiers.
            # 1. The good model.
            print 'Fake images shape:', self.fake_images.shape
            if self.mnist is True:
                self.good_predictions = self.good_model.predict(self.fake_images)
            else:
                print '\tUsing prediction prepared images.'
                self.good_predictions = self.good_model.predict(self.prediction_ready)
            self.good_accuracy = self._metric(
                self.good_labels, tf.nn.softmax(self.good_predictions))
            self.good_fn_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.good_predictions,
                    labels=self.good_labels))
            self.good_confusion = tf.confusion_matrix(
                tf.argmax(self.good_labels, axis=1),
                tf.argmax(self.good_predictions, axis=1), num_classes=2)
            # 2. The evil model.
            if self.mnist is True:
                self.evil_predictions = self.evil_model.predict(self.fake_images)
            else:
                self.evil_predictions = self.evil_model.predict(self.prediction_ready)
            self.evil_accuracy = self._metric(
                self.evil_labels, tf.nn.softmax(self.evil_predictions))
            self.evil_fn_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.evil_predictions,
                    labels=self.evil_labels))
            self.evil_confusion = tf.confusion_matrix(
                tf.argmax(self.evil_labels, axis=1),
                tf.argmax(self.evil_predictions, axis=1), num_classes=2)

            # adversarial loss = good_c * good_loss - evil_c * evil_loss.
            self.adv_loss = self.opts.good_loss_coeff * self.good_fn_loss - \
                self.opts.evil_loss_coeff * self.evil_fn_loss

            # Predict labels for images using the pre-trained model.
            # self.predict_labels = self.model.predict(self.images)
            # self.accuracy = self._metric(
            #     self.labels, tf.nn.softmax(self.predict_labels))
            # # Adding accuracy to Summary.
            # self.accuracy_sum = tf.summary.scalar("accuracy", self.accuracy)

            # self.fake_predict_labels = self.model.predict(self.fake_images)
            # self.out_predict_labels = tf.argmax(tf.nn.softmax(
            #     self.model.predict(self.fake_images_sample)), dimension=1)

            # if self.opts.is_advGAN is True:
            #     self.adv_G_loss = self._adversarial_g_loss(
            #         self.fake_predict_labels, self.labels)
            # else: # self.opts.is_advGAN == False. Using privateGAN.
            #     self.adv_G_loss = tf.reduce_mean(
            #         tf.nn.sigmoid_cross_entropy_with_logits(
            #             logits=self.fake_predict_labels,
            #             labels=self.labels))

            # self.adv_accuracy = self._metric(
            #     self.labels, tf.nn.softmax(self.fake_predict_labels))

            self.total_loss = self.gan_loss + ld * self.adv_loss
            # # Why is it only the G_loss? Why not gan_loss?
            # self.total_loss = G_lambda * G_loss + \
            #     ld * self.adv_loss # This is only added to self.gan_loss. For unknown reasons.
            #     # hinge_lambda * hinge_loss + \
            #     # L1_lambda * L1_norm + \
            #     # L2_lambda * L2_norm

            self.good_loss_sum = tf.summary.scalar("Good loss", self.good_fn_loss)
            self.evil_loss_sum = tf.summary.scalar("Evil loss", self.evil_fn_loss)
            self.adv_loss_sum = tf.summary.scalar("Adversarial loss", self.adv_loss)
            self.total_loss_sum = tf.summary.scalar("Total loss", self.total_loss)
            self.good_accuracy_sum = \
                tf.summary.scalar("Good accuracy", self.good_accuracy)
            self.evil_accuracy_sum = \
                tf.summary.scalar("Evil accuracy", self.evil_accuracy)
            # self.adv_accuracy_sum = \
            #     tf.summary.scalar("adv_accuracy", self.adv_accuracy)

            self.total_loss_merge_sum = tf.summary.merge([
                self.l1_loss_sum,
                self.l2_loss_sum,
                self.hinge_loss_sum,
                self.g_loss_sum,
                self.d_loss_sum,
                self.gan_loss_sum,
                self.good_loss_sum,
                self.evil_loss_sum,
                self.adv_loss_sum,
                self.total_loss_sum])

            t_vars = tf.trainable_variables()
            self.d_vars = [var for var in t_vars if 'd_' in var.name and 'evagan' in var.name]
            self.g_vars = [var for var in t_vars if 'g_' in var.name and 'evagan' in var.name]


            # self.d_optim = tf.train.AdamOptimizer(self.lr).minimize(self.d_loss, self.d_vars)
            D_pre_opt = tf.train.AdamOptimizer(self.lr)
            D_grads_and_vars_pre = D_pre_opt.compute_gradients(self.d_loss, self.d_vars)
            D_grads_and_vars_pre = \
                [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in D_grads_and_vars_pre]
            self.D_pre_train_op = D_pre_opt.apply_gradients(D_grads_and_vars_pre)

            # G loss without adversary loss
            # G_pre_opt = tf.train.AdamOptimizer(self.lr)
            # G_grads_and_vars_pre = G_pre_opt.compute_gradients(G_loss, self.g_vars)
            # G_grads_and_vars_pre = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in G_grads_and_vars_pre]
            # self.G_pre_train_op = G_pre_opt.apply_gradients(G_grads_and_vars_pre)
            # G loss with adversary loss
            # self.g_optim = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss, self.g_vars)
            G_opt = tf.train.AdamOptimizer(self.lr)
            G_grads_and_vars = G_opt.compute_gradients(self.total_loss, self.g_vars)
            G_grads_and_vars = \
                [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in G_grads_and_vars]
            self.G_train_op = G_opt.apply_gradients(G_grads_and_vars)

    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:
            s = 32
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv')) #28 x 28
            # h0 is (128 x 128 x self.df_dim) 16 x 16 x df_dim 14x 14
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv'))) # 14 x 14
            # h1 is (64 x 64 x self.df_dim*2) 8 x 8 x df_dim*2 7x 7
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, d_h=1, d_w=1, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)  4 x 4 x df_dim*4
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.opts.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def patch_discriminator(self, image, y=None, reuse=False):
        print "patch discriminator"
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))  #3
            # h0 is (128 x 128 x self.df_dim) 16 x 16 x df_dim 14x14
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2) 8 x 8 x df_dim*2 7x 7
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, d_h=1, d_w=1, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)  4 x 4 x df_dim*4
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            h4 = lrelu(self.d_bn5(conv2d(h3, 1, d_h=1, d_w=1, name='d_h5_conv')))

            return tf.nn.sigmoid(h4), h4

    def generator(self, image, y=None, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            if self.mnist:
                ### assume image input size is 32
                s, s2, s4, s8, s16 = 28, 14, 7, 4, 2
            else:
                s = self.opts.img_dim
                s2, s4, s8, s16, s32, s64, s128 = \
                    int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), 3, 2
                print s2, s4, s8, s16, s32, s64, s128
            # s = 32
            # # s2, s4, s8, s16, s32, s64, s128 = \
            # #   int(s/2), int(s/4), int(s/8), int(s/16), \
            # #   int(s/32), int(s/64), int(s/128)
            # s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
            # 16, 8 ,4, 2,1 s:32
            # image is (32 x 32 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (16 x 16 x self.gf_dim) 14x14
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim * 2, name='g_e2_conv'))
            # e2 is (8 x 8 x self.gf_dim*2) 7x 7
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim * 4, name='g_e3_conv'))
            # # e3 is (4 x 4 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim * 8, name='g_e4_conv'))
            # # e4 is (2 x 2 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim * 8, name='g_e5_conv'))
            # # e5 is (1 x 1 x self.gf_dim*8)

            if self.mnist is False:
                e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
                e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
                e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))

                self.d0, self.d0_w, self.d0_b = deconv2d(
                    tf.nn.relu(e8),
                    [self.batch_size, s128, s128, self.gf_dim * 8],
                    name='g_d0',
                    with_w=True)
                d0 = tf.nn.dropout(self.g_bn_d0(self.d0), 0.5)
                d0 = tf.concat([d0, e7], 3)

                self.d1, self.d1_w, self.d1_b = deconv2d(
                    tf.nn.relu(d0),
                    [self.batch_size, s64, s64, self.gf_dim * 8],
                    name='g_d1',
                    with_w=True)
                d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
                d1 = tf.concat([d1, e6], 3)

                self.d2, self.d2_w, self.d2_b = deconv2d(
                    tf.nn.relu(d1),
                    [self.batch_size, s32, s32, self.gf_dim * 8],
                    name='g_d2',
                    with_w=True)
                d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
                d2 = tf.concat([d2, e5], 3)

                self.d3, self.d3_w, self.d3_b = deconv2d(
                    tf.nn.relu(d2),
                    [self.batch_size, s16, s16, self.gf_dim * 8],
                    name='g_d3',
                    with_w=True)
                d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
                d3 = tf.concat([d3, e4], 3)

                self.d4, self.d4_w, self.d4_b = deconv2d(
                    tf.nn.relu(d3),
                    [self.batch_size, s8, s8, self.gf_dim * 4],
                    name='g_d4',
                    with_w=True)
                d4 = tf.nn.dropout(self.g_bn_d4(self.d4), 0.5)
                d4 = tf.concat([d4, e3], 3)

                self.d5, self.d5_w, self.d5_b = deconv2d(
                    tf.nn.relu(d4),
                    [self.batch_size, s4, s4, self.gf_dim * 2],
                    name='g_d5',
                    with_w=True)
                d5 = tf.nn.dropout(self.g_bn_d5(self.d5), 0.5)
                d5 = tf.concat([d5, e2], 3)

                self.d6, self.d6_w, self.d6_b = deconv2d(
                    tf.nn.relu(d5),
                    [self.batch_size, s2, s2, self.gf_dim],
                    name='g_d6',
                    with_w=True)
                d6 = self.g_bn_d6(self.d6)
                d6 = tf.concat([d6, e1], 3)

                self.d7, self.d7_w, self.d7_b = deconv2d(
                    tf.nn.relu(d6),
                    [self.batch_size, s, s, self.output_c_dim],
                    name='g_d7',
                    with_w=True)
                last_layer = self.d7

            if self.mnist is True:
                self.d1, self.d1_w, self.d1_b = deconv2d(
                    tf.nn.relu(e5),
                    [self.batch_size, s16, s16, self.gf_dim*8],
                    name='g_d1',
                    with_w=True)
                d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
                d1 = tf.concat([d1, e4], 3)
                # d1 is ( 2 x 2 x self.gf_dim*8*2)

                self.d2, self.d2_w, self.d2_b = deconv2d(
                    tf.nn.relu(d1),
                    [self.batch_size, s8, s8, self.gf_dim*4],
                    name='g_d2',
                    with_w=True)
                d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
                d2 = tf.concat([d2, e3], 3)
                # d2 is (4 x 4 x self.gf_dim*4*2)

                self.d3, self.d3_w, self.d3_b = deconv2d(
                    tf.nn.relu(d2),
                    [self.batch_size, s4, s4, self.gf_dim*2],
                    name='g_d3',
                    with_w=True)
                d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
                d3 = tf.concat([d3, e2], 3)
                # d3 is (8 x 8 x self.gf_dim*2*2)

                self.d4, self.d4_w, self.d4_b = deconv2d(
                    tf.nn.relu(d3),
                    [self.batch_size, s2, s2, self.gf_dim],
                    name='g_d4',
                    with_w=True)
                d4 = self.g_bn_d4(self.d4)
                d4 = tf.concat([d4, e1], 3)
                # d4 is (16 x 16 x self.gf_dim*8*2)

                self.d5, self.d5_w, self.d5_b = deconv2d(
                    tf.nn.relu(d4),
                    [self.batch_size, s, s, self.output_c_dim],
                    name='g_d5',
                    with_w=True)
                last_layer = self.d5

            return tf.clip_by_value(
                self.opts.c * tf.nn.tanh(last_layer) + image, -1.0, 1.0), \
                tf.nn.tanh(last_layer)

    def sampler(self, image, y=None):
        """
        A generator without the noise.
        #### Returns fake images
        """
        with tf.variable_scope("generator") as scope:

            tf.get_variable_scope().reuse_variables()
            #images = tf.image.resize_bilinear(image, [32, 32, 1])

            if self.mnist:
                ### assume image input size is 32
                s, s2, s4, s8, s16 = 28, 14, 7, 4, 2
            else:
                s = self.opts.img_dim
                s2, s4, s8, s16, s32, s64, s128 = \
                    int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), 3, 2
                print s2, s4, s8, s16, s32, s64, s128
            #16, 8 ,4, 2,1 s:32
            # image is (32 x 32 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (16 x 16 x self.gf_dim) 14x14
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (8 x 8 x self.gf_dim*2) 7x 7
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # # e3 is (4 x 4 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # # e4 is (2 x 2 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # # e5 is (1 x 1 x self.gf_dim*8)

            if self.mnist is False:
                e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
                e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
                e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))

                self.d0, self.d0_w, self.d0_b = deconv2d(
                    tf.nn.relu(e8),
                    [self.batch_size, s128, s128, self.gf_dim * 8],
                    name='g_d0',
                    with_w=True)
                d0 = tf.nn.dropout(self.g_bn_d0(self.d0), 0.5)
                d0 = tf.concat([d0, e7], 3)

                self.d1, self.d1_w, self.d1_b = deconv2d(
                    tf.nn.relu(d0),
                    [self.batch_size, s64, s64, self.gf_dim * 8],
                    name='g_d1',
                    with_w=True)
                d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
                d1 = tf.concat([d1, e6], 3)

                self.d2, self.d2_w, self.d2_b = deconv2d(
                    tf.nn.relu(d1),
                    [self.batch_size, s32, s32, self.gf_dim * 8],
                    name='g_d2',
                    with_w=True)
                d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
                d2 = tf.concat([d2, e5], 3)

                self.d3, self.d3_w, self.d3_b = deconv2d(
                    tf.nn.relu(d2),
                    [self.batch_size, s16, s16, self.gf_dim * 8],
                    name='g_d3',
                    with_w=True)
                d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
                d3 = tf.concat([d3, e4], 3)

                self.d4, self.d4_w, self.d4_b = deconv2d(
                    tf.nn.relu(d3),
                    [self.batch_size, s8, s8, self.gf_dim * 4],
                    name='g_d4',
                    with_w=True)
                d4 = tf.nn.dropout(self.g_bn_d4(self.d4), 0.5)
                d4 = tf.concat([d4, e3], 3)

                self.d5, self.d5_w, self.d5_b = deconv2d(
                    tf.nn.relu(d4),
                    [self.batch_size, s4, s4, self.gf_dim * 2],
                    name='g_d5',
                    with_w=True)
                d5 = tf.nn.dropout(self.g_bn_d5(self.d5), 0.5)
                d5 = tf.concat([d5, e2], 3)

                self.d6, self.d6_w, self.d6_b = deconv2d(
                    tf.nn.relu(d5),
                    [self.batch_size, s2, s2, self.gf_dim],
                    name='g_d6',
                    with_w=True)
                d6 = self.g_bn_d6(self.d6)
                d6 = tf.concat([d6, e1], 3)

                self.d7, self.d7_w, self.d7_b = deconv2d(
                    tf.nn.relu(d6),
                    [self.batch_size, s, s, self.output_c_dim],
                    name='g_d7',
                    with_w=True)
                last_layer = self.d7

            if self.mnist is True:
                self.d1, self.d1_w, self.d1_b = deconv2d(
                    tf.nn.relu(e5),
                    [self.batch_size, s16, s16, self.gf_dim*8],
                    name='g_d1',
                    with_w=True)
                d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
                d1 = tf.concat([d1, e4], 3)
                # d1 is ( 2 x 2 x self.gf_dim*8*2)

                self.d2, self.d2_w, self.d2_b = deconv2d(
                    tf.nn.relu(d1),
                    [self.batch_size, s8, s8, self.gf_dim*4],
                    name='g_d2',
                    with_w=True)
                d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
                d2 = tf.concat([d2, e3], 3)
                # d2 is (4 x 4 x self.gf_dim*4*2)

                self.d3, self.d3_w, self.d3_b = deconv2d(
                    tf.nn.relu(d2),
                    [self.batch_size, s4, s4, self.gf_dim*2],
                    name='g_d3',
                    with_w=True)
                d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
                d3 = tf.concat([d3, e2], 3)
                # d3 is (8 x 8 x self.gf_dim*2*2)

                self.d4, self.d4_w, self.d4_b = deconv2d(
                    tf.nn.relu(d3),
                    [self.batch_size, s2, s2, self.gf_dim],
                    name='g_d4',
                    with_w=True)
                d4 = self.g_bn_d4(self.d4)
                d4 = tf.concat([d4, e1], 3)
                # d4 is (16 x 16 x self.gf_dim*8*2)

                self.d5, self.d5_w, self.d5_b = deconv2d(
                    tf.nn.relu(d4),
                    [self.batch_size, s, s, self.output_c_dim],
                    name='g_d5',
                    with_w=True)
                last_layer = self.d5

            return tf.clip_by_value(
                self.opts.c * tf.nn.tanh(last_layer) + image, -1.0, 1.0), \
                tf.nn.tanh(last_layer)

    def generator_resnet(self, image, y=None):
        print "\tUsing ResNet Generators."

        with tf.variable_scope("generator") as scope:

            # image is 160 x 160 x input_c_dim
            s = self.opts.img_dim  # s = 160.
            # 80, 40, 20, 10, 5, 2
            s2, s4, s8, s16, s32, s64 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), 3

            def residule_block(x, dim, ks=3, s=1, name='res'):
                ##we can also try zero padding here.
                p = int((ks - 1) / 2)
                y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y = instance_norm(conv2d(y, dim, ks, ks, s, s, padding='VALID', name=name+'_c1'),
                                  name+'_bn1')
                y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y = instance_norm(conv2d(y, dim, ks, ks, s, s, padding='VALID', name=name+'_c2'),
                                  name+'_bn2')
                return y + x
            ###############
            # 160 x 160
            # image = tf.image.resize_bilinear(image, [224,224])
            e1 = tf.nn.relu(instance_norm(
                conv2d(image, self.gf_dim, k_w=7, k_h=7, d_h=1, d_w=1, name='g_e1_conv'),
                name="g_e1_conv_bn"))
            # 80 x 80
            e2 = tf.nn.relu(instance_norm(
                conv2d(e1, self.gf_dim *2, k_w=3, k_h=3, d_h=2, d_w=2, name='g_e2_conv'),
                name="g_e2_conv_bn"))
            # 40 x 40
            e3 = tf.nn.relu(instance_norm(
                conv2d(e2, self.gf_dim *4, k_w=3, k_h=3, d_h=2, d_w=2, name='g_e3_conv'),
                name="g_e3_conv_bn"))
            # 20 x 20
            e4 = tf.nn.relu(instance_norm(
                conv2d(e3, self.gf_dim *8, k_w=3, k_h=3, d_h=2, d_w=2, name='g_e4_conv'),
                name="g_e4_conv_bn"))
            # 10 x 10
            e5 = tf.nn.relu(instance_norm(
                conv2d(e4, self.gf_dim *8, k_w=3, k_h=3, d_h=2, d_w=2, name='g_e5_conv'),
                name="g_e5_conv_bn"))
            # 5 x 5
            e6 = tf.nn.relu(instance_norm(
                conv2d(e5, self.gf_dim *8, k_w=3, k_h=3, d_h=2, d_w=2, name='g_e6_conv'),
                name="g_e6_conv_bn"))
            # 2 x 2
            e7 = tf.nn.relu(instance_norm(
                conv2d(e6, self.gf_dim *8, k_w=3, k_h=3, d_h=2, d_w=2, name='g_e7_conv'),
                name="g_e7_conv_bn"))
            #2x3
            # e8 = instance_norm(
            #     conv2d(e7, self.gf_dim*8, k_w=3, k_h=3, name='g_e8_conv'), "g_e8_conv_bn")
            # st()
            #7X7
            r1 = residule_block(e7, self.gf_dim * 8, name="g_r1")
            r2 = residule_block(r1, self.gf_dim * 8, name="g_r2")
            r3 = residule_block(r2, self.gf_dim * 8, name="g_r3")
            r4 = residule_block(r3, self.gf_dim * 8, name="g_r4")

            # r5 = residule_block(r4, self.gf_dim * 8, name = "g_r5")
            # r6 = residule_block(r5, self.gf_dim * 8, name = "g_r6")
            # r7 = residule_block(r6, self.gf_dim * 8, name = "g_r7")
            # r8 = residule_block(r7, self.gf_dim * 8, name = "g_r8")


            # r9 = residule_block(r8, self.gf_dim * 8, name = "g_r9")
            # r10 = residule_block(r9, self.gf_dim * 8, name = "g_r10")
            # r11 = residule_block(r10, self.gf_dim * 8, name = "g_r11")
            # r12 = residule_block(r11, self.gf_dim * 8, name = "g_r12")
            #####
            d1 = deconv2d(r4, [self.batch_size, s64, s64, self.gf_dim*8],
                          k_h=3, k_w=3, name="g_d1", with_w=False)
            d1 = tf.nn.relu(instance_norm(d1, "g_d1_bn"))
            # [d1, e7]
            d2 = deconv2d(d1, [self.batch_size, s32, s32, self.gf_dim * 8],
                          k_h=3, k_w=3, name="g_d2", with_w=False)
            d2 = tf.nn.relu(instance_norm(d2, "g_d2_bn"))

            d3 = deconv2d(d2, [self.batch_size, s16, s16, self.gf_dim * 8],
                          k_h=3, k_w=3, name="g_d3", with_w=False)
            d3 = tf.nn.relu(instance_norm(d3, "g_d3_bn"))

            d4 = deconv2d(d3, [self.batch_size, s8, s8, self.gf_dim * 8],
                          k_h=3, k_w=3, name="g_d4", with_w=False)
            d4 = tf.nn.relu(instance_norm(d4, "g_d4_bn"))

            d5 = deconv2d(d4, [self.batch_size, s4, s4, self.gf_dim * 4],
                          k_h=3, k_w=3, name="g_d5", with_w=False)
            d5 = tf.nn.relu(instance_norm(d5, "g_d5_bn"))

            d6 = deconv2d(d5, [self.batch_size, s2, s2, self.gf_dim * 2],
                          k_h=3, k_w=3, name="g_d6", with_w=False)
            d6 = tf.nn.relu(instance_norm(d6, "g_d6_bn"))

            d7 = deconv2d(d6, [self.batch_size, s, s, self.gf_dim],
                          k_h=3, k_w=3, name="g_d7", with_w=False)
            d7 = tf.nn.relu(instance_norm(d7, "g_d7_bn"))

            d8 = deconv2d(d7, [self.batch_size, s, s, self.output_c_dim],
                          k_h=7, k_w=7, d_h=1, d_w=1, name="g_d8", with_w=False)
            return tf.clip_by_value(self.opts.c * tf.nn.tanh(d8) + image, -1.0, 1.0),\
                tf.nn.tanh(d8)

    def sampler_resnet(self, image, y=None):
        print "resnet Generator."

        with tf.variable_scope("generator") as scope:
            tf.get_variable_scope().reuse_variables()

            # image is 160 x 160 x input_c_dim
            s = self.opts.img_dim  # s = 160.
            # 80, 40, 20, 10, 5, 2
            s2, s4, s8, s16, s32, s64 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), 3

            def residule_block(x, dim, ks=3, s=1, name='res'):
                p = int((ks - 1) / 2)
                y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y = instance_norm(conv2d(y, dim, ks, ks, s, s, padding='VALID', name=name+'_c1'),
                                  name+'_bn1')
                y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y = instance_norm(conv2d(y, dim, ks, ks, s, s, padding='VALID', name=name+'_c2'),
                                  name+'_bn2')
                return y + x
            ###############
            # 160 x 160
            # image = tf.image.resize_bilinear(image, [224,224])
            e1 = tf.nn.relu(instance_norm(
                conv2d(image, self.gf_dim, k_w=7, k_h=7, d_h=1, d_w=1, name='g_e1_conv'),
                name="g_e1_conv_bn"))
            # 80 x 80
            e2 = tf.nn.relu(instance_norm(
                conv2d(e1, self.gf_dim *2, k_w=3, k_h=3, d_h=2, d_w=2, name='g_e2_conv'),
                name="g_e2_conv_bn"))
            # 40 x 40
            e3 = tf.nn.relu(instance_norm(
                conv2d(e2, self.gf_dim *4, k_w=3, k_h=3, d_h=2, d_w=2, name='g_e3_conv'),
                name="g_e3_conv_bn"))
            # 20 x 20
            e4 = tf.nn.relu(instance_norm(
                conv2d(e3, self.gf_dim *8, k_w=3, k_h=3, d_h=2, d_w=2, name='g_e4_conv'),
                name="g_e4_conv_bn"))
            # 10 x 10
            e5 = tf.nn.relu(instance_norm(
                conv2d(e4, self.gf_dim *8, k_w=3, k_h=3, d_h=2, d_w=2, name='g_e5_conv'),
                name="g_e5_conv_bn"))
            # 5 x 5
            e6 = tf.nn.relu(instance_norm(
                conv2d(e5, self.gf_dim *8, k_w=3, k_h=3, d_h=2, d_w=2, name='g_e6_conv'),
                name="g_e6_conv_bn"))
            # 2 x 2
            e7 = tf.nn.relu(instance_norm(
                conv2d(e6, self.gf_dim *8, k_w=3, k_h=3, d_h=2, d_w=2, name='g_e7_conv'),
                name="g_e7_conv_bn"))
            #2x3
            # e8 = instance_norm(
            #     conv2d(e7, self.gf_dim*8, k_w=3, k_h=3, name='g_e8_conv'), "g_e8_conv_bn")
            # st()
            #7X7
            r1 = residule_block(e7, self.gf_dim * 8, name="g_r1")
            r2 = residule_block(r1, self.gf_dim * 8, name="g_r2")
            r3 = residule_block(r2, self.gf_dim * 8, name="g_r3")
            r4 = residule_block(r3, self.gf_dim * 8, name="g_r4")

            # r5 = residule_block(r4, self.gf_dim * 8, name = "g_r5")
            # r6 = residule_block(r5, self.gf_dim * 8, name = "g_r6")
            # r7 = residule_block(r6, self.gf_dim * 8, name = "g_r7")
            # r8 = residule_block(r7, self.gf_dim * 8, name = "g_r8")


            # r9 = residule_block(r8, self.gf_dim * 8, name = "g_r9")
            # r10 = residule_block(r9, self.gf_dim * 8, name = "g_r10")
            # r11 = residule_block(r10, self.gf_dim * 8, name = "g_r11")
            # r12 = residule_block(r11, self.gf_dim * 8, name = "g_r12")
            #####
            d1 = deconv2d(r4, [self.batch_size, s64, s64, self.gf_dim*8],
                          k_h=3, k_w=3, name="g_d1", with_w=False)
            d1 = tf.nn.relu(instance_norm(d1, "g_d1_bn"))
            # [d1, e7]
            d2 = deconv2d(d1, [self.batch_size, s32, s32, self.gf_dim * 8],
                          k_h=3, k_w=3, name="g_d2", with_w=False)
            d2 = tf.nn.relu(instance_norm(d2, "g_d2_bn"))

            d3 = deconv2d(d2, [self.batch_size, s16, s16, self.gf_dim * 8],
                          k_h=3, k_w=3, name="g_d3", with_w=False)
            d3 = tf.nn.relu(instance_norm(d3, "g_d3_bn"))

            d4 = deconv2d(d3, [self.batch_size, s8, s8, self.gf_dim * 8],
                          k_h=3, k_w=3, name="g_d4", with_w=False)
            d4 = tf.nn.relu(instance_norm(d4, "g_d4_bn"))

            d5 = deconv2d(d4, [self.batch_size, s4, s4, self.gf_dim * 4],
                          k_h=3, k_w=3, name="g_d5", with_w=False)
            d5 = tf.nn.relu(instance_norm(d5, "g_d5_bn"))

            d6 = deconv2d(d5, [self.batch_size, s2, s2, self.gf_dim * 2],
                          k_h=3, k_w=3, name="g_d6", with_w=False)
            d6 = tf.nn.relu(instance_norm(d6, "g_d6_bn"))

            d7 = deconv2d(d6, [self.batch_size, s, s, self.gf_dim],
                          k_h=3, k_w=3, name="g_d7", with_w=False)
            d7 = tf.nn.relu(instance_norm(d7, "g_d7_bn"))

            d8 = deconv2d(d7, [self.batch_size, s, s, self.output_c_dim],
                          k_h=7, k_w=7, d_h=1, d_w=1, name="g_d8", with_w=False)
            return tf.clip_by_value(self.opts.c * tf.nn.tanh(d8) + image, -1.0, 1.0),\
                tf.nn.tanh(d8)
