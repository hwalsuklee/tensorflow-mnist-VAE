import tensorflow as tf
import numpy as np
import mnist_data
import os
import time
import vae
import plot_utils
import glob

import argparse

DIM1_SIZE_MNIST = 28

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of 'Variational AutoEncoder (VAE)'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--results_path', type=str, default='results',
                        help='File path of output images')

    parser.add_argument('--add_noise', type=bool, default=False, help='Boolean for adding salt & pepper noise to input image')

    parser.add_argument('--dim_z', type=int, default='20', help='Dimension of latent vector', required = True)

    parser.add_argument('--network_type', type=str, default='CNN', choices=['CNN','MLP'], help='Network type for VAE. choose between CNN and MLP', required = True)

    parser.add_argument('--n_hidden', type=int, default=500, help='number of hidden units in MLP')

    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=10, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')

    parser.add_argument('--PRP_FLAG', type=bool, default=True,
                        help='Boolean for plot-reproduce-performance')

    parser.add_argument('--PRP_n_img_x', type=int, default=8,
                        help='Number of images along x-axis')

    parser.add_argument('--PRP_n_img_y', type=int, default=8,
                        help='Number of images along y-axis')

    parser.add_argument('--PRP_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PMLR_FLAG', type=bool, default=False,
                        help='Boolean for plot-manifold-learning-result')

    parser.add_argument('--PMLR_n_img_x', type=int, default=20,
                        help='Number of images along x-axis')

    parser.add_argument('--PMLR_n_img_y', type=int, default=20,
                        help='Number of images along y-axis')

    parser.add_argument('--PMLR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PMLR_z_range', type=float, default=3.0,
                        help='Range for unifomly distributed latent vector')

    parser.add_argument('--PMLR_n_samples', type=int, default=3000,
                        help='Number of samples in order to get distribution of labeled data')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):

    # --results_path
    try:
        os.mkdir(args.results_path)
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(args.results_path+'/*')
    for f in files:
        os.remove(f)

    # --add_noise
    try:
        assert args.add_noise == True or args.add_noise == False
    except:
        print('add_noise must be boolean type')
        return None

    # --dim-z
    try:
        assert args.dim_z > 0
    except:
        print('dim_z must be positive integer')
        return None

    # --network_type
    try:
        assert args.network_type == 'CNN' or args.network_type == 'MLP'
    except:
        print('network_type must be one of {CNN, MLP}')
        return None

    # --n_hidden
    if args.network_type == 'MLP':
        try:
            assert args.n_hidden >= 1
        except:
            print('number of hidden units must be larger than one')
            return None

    # --learn_rate
    try:
        assert args.learn_rate > 0
    except:
        print('learning rate must be positive')
        return None

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')
        return None

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
        return None

    # --PRP_FLAG
    try:
        assert args.PRP_FLAG == True or args.PRP_FLAG == False
    except:
        print('PRP_FLAG must be boolean type')
        return None

    if args.PRP_FLAG == True:
        # --PRP_n_img_x, --PRP_n_img_y
        try:
            assert args.PRP_n_img_x >= 1 and args.PRP_n_img_y >= 1
        except:
            print('PRP : number of images along each axis must be larger than or equal to one')
            return None

        # --PRP_resize_factor
        try:
            assert args.PRP_resize_factor > 0
        except:
            print('PRP : resize factor for each displayed image must be positive')
            return None

    # --PMLR_FLAG
    try:
        assert args.PMLR_FLAG == True or args.PMLR_FLAG == False
    except:
        print('PMLR_FLAG must be boolean type')
        return None

    if args.PMLR_FLAG == True:
        try:
            assert args.dim_z == 2
        except:
            print('PMLR : dim_z must be two')
            return None

        # --PMLR_n_img_x, --PMLR_n_img_y
        try:
            assert args.PMLR_n_img_x >= 1 and args.PMLR_n_img_y >= 1
        except:
            print('PMLR : number of images along each axis must be larger than or equal to one')
            return None

        # --PMLR_resize_factor
        try:
            assert args.PMLR_resize_factor > 0
        except:
            print('PMLR : resize factor for each displayed image must be positive')
            return None

        # --PMLR_z_range
        try:
            assert args.PMLR_z_range > 0
        except:
            print('PMLR : range for unifomly distributed latent vector must be positive')
            return None

        # --PMLR_n_samples
        try:
            assert args.PMLR_n_samples > 100
        except:
            print('PMLR : Number of samples in order to get distribution of labeled data must be large enough')
            return None

    return args

"""main function"""
def main(args):

    """ parameters """
    RESULTS_DIR = args.results_path

    # network architecture
    ADD_NOISE = args.add_noise

    n_hidden = args.n_hidden
    dim_img = DIM1_SIZE_MNIST**2  # number of pixels for a MNIST image
    dim_z = args.dim_z

    # train
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    learn_rate = args.learn_rate

    # Plot
    PRP_FLAG = args.PRP_FLAG    # Plot Reproduce Performance
    PMLR_FLAG = args.PMLR_FLAG  # Plot Manifold Learning Result
    
    PRP_n_img_x = args.PRP_n_img_x
    PRP_n_img_y = args.PRP_n_img_y
    PRP_resize_factor = args.PRP_resize_factor

    PMLR_n_img_x = args.PMLR_n_img_x
    PMLR_n_img_y = args.PMLR_n_img_y
    PMLR_resize_factor = args.PMLR_resize_factor
    PMLR_z_range = args.PMLR_z_range
    PMLR_n_samples = args.PMLR_n_samples

    """ prepare MNIST data """

    train_total_data, train_size, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()
    n_samples = train_size

    """ build graph """

    # input placeholders
    # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
    x_hat = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
    x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')

    # input for PMLR
    z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')

    # network architecture
    y, z, loss, neg_marginal_likelihood, KL_divergence = vae.autoencoder(x_hat, x, dim_img, dim_z, n_hidden, TYPE='CNN')

    # optimization
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    """ training """

    # Plot for reproduce performance
    if PRP_FLAG:
        PRP = plot_utils.Plot_Reproduce_Performance(RESULTS_DIR, PRP_n_img_x, PRP_n_img_y, DIM1_SIZE_MNIST, DIM1_SIZE_MNIST, PRP_resize_factor)

        x_PRP = test_data[0:PRP.n_tot_imgs, :]

        x_PRP_img = x_PRP.reshape(PRP.n_tot_imgs, DIM1_SIZE_MNIST, DIM1_SIZE_MNIST)
        PRP.save_images(x_PRP_img, name='input.jpg')

    # Plot for manifold learning result
    if PMLR_FLAG and dim_z == 2:

        PMLR = plot_utils.Plot_Manifold_Learning_Result(RESULTS_DIR, PMLR_n_img_x, PMLR_n_img_y, DIM1_SIZE_MNIST, DIM1_SIZE_MNIST, PMLR_resize_factor, PMLR_z_range)

        x_PMLR = test_data[0:PMLR_n_samples, :]
        id_PMLR = test_labels[0:PMLR_n_samples, :]

        if ADD_NOISE:
            x_PMLR = x_PMLR * np.random.randint(2, size=x_PMLR.shape)
            x_PMLR += np.random.randint(2, size=x_PMLR.shape)

        decoded = vae.decoder(z_in, dim_img, n_hidden, TYPE='CNN')


    # add noise
    if PRP_FLAG and ADD_NOISE:
        x_PRP = x_PRP * np.random.randint(2, size=x_PRP.shape)
        x_PRP += np.random.randint(2, size=x_PRP.shape)

        x_PRP_img = x_PRP.reshape(PRP.n_tot_imgs, DIM1_SIZE_MNIST, DIM1_SIZE_MNIST)
        PRP.save_images(x_PRP_img, name='input_noise.jpg')

    # train
    total_batch = int(n_samples / batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for epoch in range(n_epochs):

            # Random shuffling
            np.random.shuffle(train_total_data)
            train_data_ = train_total_data[:, :-mnist_data.NUM_LABELS]
            train_labels_ = train_total_data[:, -mnist_data.NUM_LABELS:]

            # Loop over all batches
            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % (n_samples)
                batch_xs_input = train_data_[offset:(offset + batch_size), :]

                batch_xs_target = batch_xs_input

                # add salt & pepper noise
                if ADD_NOISE:
                    batch_xs_input = batch_xs_input * np.random.randint(2, size=batch_xs_input.shape)
                    batch_xs_input += np.random.randint(2, size=batch_xs_input.shape)

                _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                    (train_op, loss, neg_marginal_likelihood, KL_divergence),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target})

            # print cost every epoch
            print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (epoch, tot_loss, loss_likelihood, loss_divergence))

            # Plot for reproduce performance
            if PRP_FLAG:
                y_PRP = sess.run(y, feed_dict={x_hat: x_PRP})
                y_PRP_img = y_PRP.reshape(PRP.n_tot_imgs, DIM1_SIZE_MNIST, DIM1_SIZE_MNIST)
                PRP.save_images(y_PRP_img, name="/PRP_epoch_%02d" %(epoch) + ".jpg")

            # Plot for manifold learning result
            if PMLR_FLAG and dim_z == 2:
                y_PMLR = sess.run(decoded, feed_dict={z_in: PMLR.z})
                y_PMLR_img = y_PMLR.reshape(PMLR.n_tot_imgs, DIM1_SIZE_MNIST, DIM1_SIZE_MNIST)
                PMLR.save_images(y_PMLR_img, name="/PMLR_epoch_%02d" % (epoch) + ".jpg")

                # plot distribution of labeled images
                z_PMLR = sess.run(z, feed_dict={x_hat: x_PMLR})
                PMLR.save_scattered_image(z_PMLR,id_PMLR, name="/PMLR_scatterd_epoch_%02d" % (epoch) + ".jpg")

        end_time = time.time()
        print('Total processing time : %g' % (end_time - start_time))


if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # main
    main(args)