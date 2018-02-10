"""
Module for argument parsing.
"""
import argparse


def parse_opt():
    """
    Parses experiment options.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_path', type=str, default='MNIST_data/perturbed',
                        help='Path for the output perturbed data.')

    # whitebox
    parser.add_argument('--model_restore', type=str, default="models/mnist",
                        help="checkpoint model name of whitebox")
    parser.add_argument('--good_model_path', type=str,
                        default='models/A_odd_even',
                        help='path for saved GOOD model')
    parser.add_argument('--evil_model_path', type=str,
                        default='models/A_digits',
                        help='path for saved EVIL model')

    parser.add_argument('--evil_label_num', type=int, default=10,
                        help='Number of labels in the adversarial classifier')
    parser.add_argument('--good_label_num', type=int, default=2,
                        help='Number of labels in the safe classifier')

    parser.add_argument('--bound', type=float, default=0.6, help="bound")

    # flag:
    parser.add_argument('--iter_model', type=int, default=0,
                        help="random ensemble")

    parser.add_argument('--rw', type=int, default=0, help="random ensemble")

    parser.add_argument('--linf_flag', type=int, default=0, help="linf")
    parser.add_argument('--ensemble', type=int, default=0, help="ensemble or not")
    parser.add_argument('--cgan_flag', type=int, default=1, help="cgan_flag")
    parser.add_argument('--patch_flag', type=int, default=1, help="patch_flag")

    parser.add_argument('--hinge_flag', type=int, default=0, help="hinge_flag")

    parser.add_argument('--prefix', type=str, default="")
    # loss params

    parser.add_argument(
        '--linf_lambda', type=float, default=0, help='linf_lambda')
    parser.add_argument(
        '--H_lambda', type=float, default=0, help='Hinge loss lambda')

    parser.add_argument('--ld', type=float, default=1.0,
                        help='adversarial loss coefficient')
    parser.add_argument('--L1_lambda', type=float, default=0,
                        help='L1 loss coefficient')
    parser.add_argument('--L2_lambda', type=float, default=0,
                        help='L2 loss coefficient')
    parser.add_argument('--G_lambda', type=float, default=0,
                        help='Generator loss coefficient')
    parser.add_argument('--c', type=float, default=1,
                        help='c')
    parser.add_argument('--evil_loss_coeff', type=float, default=.75,
                        help='Coefficient for the evil classifier loss value.')
    parser.add_argument('--good_loss_coeff', type=float, default=1.0,
                        help='Coefficient for the good classifier loss value.')

    parser.add_argument('--s_l', type=int, default=0, help="source_label")
    parser.add_argument('--t_l', type=int, default=1, help="target_label")

    parser.add_argument('--model_name', type=str, default='MLP',
                        help="load_model_name")
    parser.add_argument('--add_neg_every', type=int, default=5,
                        help="add negative samples from generator")
    parser.add_argument('--add_neg_iteration', type=int, default=10000,
                        help="add negative samples from generator")
    parser.add_argument('--input_data', type=str, default="MNIST",
                        help="input data MNIST or CIFAR10 ")
    parser.add_argument('--train_adv', type=int, default=0,
                        help="using adverarial loss or not ")
    parser.add_argument('--pretrain_iteration', type=int, default=3000,
                        help="pretrain iteration")
    parser.add_argument('--max_iteration', type=int, default=2000,
                        help="Maximum training iteration (default: 2000).")

    parser.add_argument('--confidence', type=float, default=0.0,
                        help="confidence")

    parser.add_argument('--h_dim', type=int, default=128,
                        help='hidden_dim')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch_size')
    parser.add_argument('--input_c_dim', type=int, default=1,
                        help='input_channel_dim')
    parser.add_argument('--output_c_dim', type=int, default=1,
                        help='output_channel_dim')
    parser.add_argument('--gf_dim', type=int, default=8,
                        help='generator_filter_dim')
    parser.add_argument('--df_dim', type=int, default=8,
                        help='discriminator_filter_dim')
    parser.add_argument('--fine_tune', type=int, default=0,
                        help='fine_tune(0:no, 1:yes)')
    parser.add_argument('--img_dim', type=int, default=28,
                        help='image_w_h_dim')


    # parser.add_argument('--log_path', type=str, default = 'log_path',
    #                 help='log_path')
    parser.add_argument('--image_path', type=str, default='./lfw_data/',
                        help='image_path')
    parser.add_argument('--load_checkpoint_path', type=str, default='GAN/save',
                        help='directory to store checkpointed models')
    parser.add_argument('--checkpoint_path', type=str, default='GAN/save',
                        help='directory to store checkpointed models')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1,
                        help='at what iteration to start decaying learning rate?' + \
                             '(-1 = dont) (in epoch)')
    parser.add_argument('--decay_iteration_max', type=int, default=50000,
                        help='decay iteration max')
    parser.add_argument('--learning_rate_decay_every', type=int, default=20000,
                        help='every how many iterations thereafter to drop LR by half?(in epoch)')
    parser.add_argument('--save_checkpoint_every', type=int, default=200,
                        help='how often to save a model checkpoint (in iterations)')
    parser.add_argument('--losses_log_every', type=int, default=100,
                        help='How often do we snapshot losses, for inclusion in the' + \
                             'progress dump? (0 = disable)')
    parser.add_argument('--is_advGAN', type=bool, default=False,
                        help='Determines whether or not we are using advGAN. \
                        If False, we are using privateGAN.')

    parser.add_argument('--cw', type=int, default=1,
                        help='c&w loss or not')

    parser.add_argument('--targeted', type=int, default=1,
                        help='c&w loss or not')

    # LFW Arguments
    parser.add_argument('--lfw_input_path', type=str, default='./lfw_data/',
                        help='Path to LFW data.')
    parser.add_argument('--gender_model_path', type=str, default='./models/lfw_gender',
                        help='Path to LFW data.')
    parser.add_argument('--id_model_path', type=str, default='./models/lfw_id',
                        help='Path to LFW data.')
    parser.add_argument('--lfw_base_path', type=str, default='./models/lfw',
                        help='Path to LFW two models.')
    parser.add_argument('--resnet_gen', dest='resnet_gen', action='store_true',
                        help='Uses ResNet Generators.')
    parser.add_argument('--no_resnet_gen', dest='resnet_gen', action='store_false',
                        help='Does not use ResNet Generators.')
    parser.set_defaults(resnet_gen=False)
    parser.add_argument('--balance_data', dest='balance_data', action='store_true',
                        help='Balance dataset.')
    parser.add_argument('--no_balance_data', dest='balance_data', action='store_false',
                        help='Does not balance dataset.')
    parser.set_defaults(balance_data=True)
    parser.add_argument('--balance_ratio', type=float, default=2,
                        help='Balancing ratio.')
    parser.add_argument('--balance_gender', dest='balance_gender', action='store_true',
                        help='Balance genders in the dataset.')
    parser.add_argument('--no_balance_gender', dest='balance_gender', action='store_false',
                        help='Does not balance genders in the dataset.')
    parser.set_defaults(balance_gender=True)

    args = parser.parse_args()

    return args
