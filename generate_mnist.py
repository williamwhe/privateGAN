
import math
import numpy as np 
import tensorflow as tf
from ops import *
import os
import time
import scipy.io  as sio
import random
from tensorflow.examples.tutorials.mnist import input_data
import pdb
import opts 
from utils import plot
from advgan import advGAN
import cifar10
from utils import save_images
from Dataset2 import Dataset2 

from setup_mnist import MNIST, MNISTModel, MNISTModel2, MNISTModel3
from setup_cifar import CIFARModel, CIFARModel2, CIFARModel3

tag_num = 0
def train():
    flatten_flag = True ### the output of G need to flatten or not?
    opt = opts.parse_opt()
    opt.input_data = "MNIST"
    # mapping [0,1] -> [-1,1]
    # load data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_data =  mnist.train.images * 2.0 - 1.0
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
    tf_config.intra_op_parallelism_threads=NUM_THREADS
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        # Initialize the variables, and restore the variables form checkpoint if there is.
        # and initialize the writer
        iteration = 0
        sample_matrix = []
        source_matrix = []
        magnitude_matrix = []
        min_adv_accuracys = []
        corsp_magnitudes = []
        #initial whitebox model
        model_store = opt.model_restore
        whitebox_model = MNISTModel(model_store, sess)
        #initial ADVGAN
        model = advGAN(whitebox_model, model_store, opt, sess )

        iteration = 0
        min_adv_accuracy = 10e10
        source_label =  opt.s_l
        target_label = opt.t_l
        if source_label == target_label:
            print("source label equals to target label")
            return
        print("source{}, target{} .".format(source_label, target_label))
        writer = tf.summary.FileWriter("logs", sess.graph)
        loader = Dataset2(train_data, train_label, source_label)
        model_num = 0
        
        best_show_samples_magintude = []
        best_show_samples = []
        best_show_source_imgs = []
        best_show_idxs = []
        while iteration < 2000:
            # pass
            data = loader.next_batch(batch_size, negative = False ) 
            labels = np.zeros_like( data[1] )
            labels[:, target_label] = 1

            feed = {model.source :  data[0], \
                model.labels: labels,\
                model.target : data[2]}

            summary_str, G_loss, pre_G_loss, adv_G_loss, L1_norm, L2_norm, hinge_loss, _ = sess.run( [ model.g_loss_add_adv_merge_sum, model.G_loss_add_adv, model.pre_G_loss, model.adv_G_loss , model.L1_norm , model.L2_norm , model.hinge_loss, model.G_train_op ], feed)
            writer.add_summary(summary_str, iteration)

            summary_str, D_loss, _ = sess.run( [model.pre_d_loss_sum, model.D_loss, model.D_pre_train_op], feed )
            writer.add_summary(summary_str, iteration)

            if iteration != 0 and iteration % opt.losses_log_every == 0:    
                print("loss(D, G, pre_G_loss,  adv_G, L1_norm, L2_norm, hinge_loss ): ", D_loss, G_loss, pre_G_loss,  adv_G_loss, L1_norm, L2_norm, hinge_loss)
                print("iteration: ", iteration)


            if (iteration != 0 and iteration % opt.save_checkpoint_every == 0):
            # if (iteration != 0 and iteration % 500 == 0):
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.ckpt')
                                
                # model.saver.save(sess, checkpoint_path, global_step = iteration)   

                test_loader = Dataset2(test_data, test_label, source_label)

                test_num = test_loader._num_examples
                test_iter_num = int((test_num - batch_size )  / batch_size)
                # test_iter_num = 1
                test_acc = 0.0
                test_adv_acc = 0.0
                show_samples = []

                save_samples = []

                for i in range(test_iter_num):

                    s_imgs, s_label, _  = test_loader.next_batch(batch_size, negative = False)

                    # s_imgs, s_label, _  = loader.next_batch(batch_size, negative = False)

                    feed = {model.source : s_imgs,  model.labels : s_label}

                    test_accuracy, test_adv_accuracy = sess.run( [model.accuracy, model.adv_accuracy], feed)
                    test_acc += test_accuracy
                    test_adv_acc += test_adv_accuracy
                    
                    feed = {model.source : s_imgs}
                    samples, out_predict_labels  = sess.run([ model.fake_images_sample, model.out_predict_labels], feed)
                    idxs = np.where(out_predict_labels == target_label)[0]
                    # save_images(samples[:100], [10, 10], 'CIFAR10/result2/test_' + str(source_idx) + str(target_idx)+  '_.png')                
                    # pdb.set_trace()
                    show_samples.append(samples)
                    save_samples.append(samples[idxs])
                show_samples = np.concatenate(show_samples, axis = 0)
                save_samples = np.concatenate(save_samples, axis = 0)
                test_accuracy = test_acc / float( test_iter_num )
                test_adv_accuracy = test_adv_acc / float( test_iter_num )
                if min_adv_accuracy > test_adv_accuracy:
                    min_adv_accuracy = test_adv_accuracy
                    test_out_str =  "test total accuracy {}, adv accuracy {}".format( str( test_accuracy ) ,str( test_adv_accuracy ) )   
                    print(test_out_str)
                    save_images(save_samples[:100], [10, 10], 'result.png')
            iteration += 1



        #########################         
               
if __name__ == "__main__":
    train()
