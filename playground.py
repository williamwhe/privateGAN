"""
This is a playground file, including tests and experiments.
"""

import numpy as np
import opts
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from setup_mnist import MNISTModel
from advgan import advGAN
import matplotlib.pyplot as plt

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_data = mnist.train.images * 2.0 - 1.0
    train_label = mnist.train.labels

    test_data = mnist.test.images * 2.0 - 1.0
    test_label = mnist.test.labels

    x_dim = train_data.shape[1]
    y_dim = train_label.shape[1]

    opt = opts.parse_opt()
    batch_size = opt.batch_size

    # Changing the options here.
    opt.input_data = "MNIST"
    opt.input_c_dim = 1
    opt.output_c_dim = 1
    opt.input_dim = x_dim
    opt.label_dim = y_dim
    # Running arguments
    opt.c = 1.
    opt.ld = 500.
    opt.H_lambda = 10.
    opt.cgan_flag = True
    opt.patch_flag = True
    opt.G_lambda = 10.
    opt.s_l = 0
    opt.t_l = 1

    # batch_size = opt.batch_size


    # Runnign a session, to load the saved model.
    with tf.Session() as sess:
        model_store = opt.model_restore
        print 'MNIST model is stored at %s' % model_store
        whitebox_model = MNISTModel(model_store)
        #initial ADVGAN
        model = advGAN(whitebox_model, model_store, opt, sess )

        best_model_path = './GAN/save/best.ckpt'
        print 'advGAN is stored at %s' % best_model_path
        model.load(best_model_path)

        # tvars = tf.trainable_variables()
        # tvars_vals = sess.run(tvars)

        # for var, val in zip(tvars, tvars_vals):
        #     if 'generator' not in var.name:
        #         continue
        #     print(var.name, val.shape)  # Prints the name of the variable alongside its value.

        # We have to load a batch of images, then create the fake ones.
        # They should be identical.
        num_images = 10
        images = train_data[:num_images]
        fake_images = sess.run([model.fake_images_sample],
                               {model.source: images})

        plt.imshow(np.reshape(fake_images[0], [28, 28]))
        plt.show()

if __name__ == '__main__':
    main()
