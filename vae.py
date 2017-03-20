import tensorflow as tf

# leaky reLu unit
def leaky_relu(x, leakiness=0.2, name="leaky_relu"):
    with tf.variable_scope(name):
        return tf.select(tf.less(x, 0.0), leakiness * x, x)

# Gaussian CNN as encoder
def gaussian_CNN_encoder(x, n_output, is_identity_cov=False):
    w_init = tf.truncated_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    with tf.variable_scope("gaussian_CNN_encoder"):
        x = tf.reshape(x, [-1, 28, 28, 1])

        n_filter1 = 16
        n_filter2 = n_filter1*2

        # 1st conv layer
        w0 = tf.get_variable('conv_w0', [5, 5, 1, n_filter1], initializer=w_init)
        b0 = tf.get_variable('conv_b0', [n_filter1], initializer=b_init)
        h0 = leaky_relu(tf.nn.conv2d(x, w0, strides=[1, 2, 2, 1], padding="SAME") + b0)

        # 2nd conv layer
        w1 = tf.get_variable('conv_w1', [5, 5, n_filter1, n_filter2], initializer=w_init)
        b1 = tf.get_variable('conv_b1', [n_filter2], initializer=b_init)
        h1 = leaky_relu(tf.nn.conv2d(h0, w1, strides=[1, 2, 2, 1], padding="SAME") + b1)

        h1 = tf.reshape(h1, [-1, 7*7*n_filter2])

        # output layer-mean
        w2 = tf.get_variable('fc_w2', [h1.get_shape()[1], n_output], initializer=w_init)
        b2 = tf.get_variable('fc_b2', [n_output], initializer=b_init)
        mean = tf.matmul(h1, w2) + b2

        if is_identity_cov:
            return mean

        # output layer-stddev
        w3 = tf.get_variable('fc_w3', [h1.get_shape()[1], n_output], initializer=w_init)
        b3 = tf.get_variable('fc_b3', [n_output], initializer=b_init)
        stddev = tf.matmul(h1, w3) + b3

        return mean, stddev

# Bernoulli CNN as decoder
def bernoulli_CNN_decoder(z, reuse=False):
    w_init = tf.truncated_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(0.0)

    with tf.variable_scope("bernoulli_CNN_decoder", reuse=reuse):

        n_filter1 = 32
        n_filter2 = int(n_filter1/2)

        # 1st fully-connected layer
        w0 = tf.get_variable('fc_w0', [z.get_shape()[1], 7*7*n_filter1], initializer=w_init)
        b0 = tf.get_variable('fc_b0', [7*7*n_filter1], initializer=b_init)
        h0 = tf.nn.relu(tf.matmul(z, w0) + b0)

        h0 = tf.reshape(h0, [-1, 7, 7, n_filter1])

        # 2nd conv_transpose layer
        w1 = tf.get_variable('convt_w1', [5, 5, n_filter2, n_filter1], initializer=w_init)
        b1 = tf.get_variable('convt_b1', [n_filter2], initializer=b_init)
        h1 = tf.nn.relu(tf.nn.conv2d_transpose(h0, w1, output_shape=[tf.shape(h0)[0], 14, 14, n_filter2], strides=[1, 2, 2, 1])+b1)

        # 3nd conv_transpose layer
        w2 = tf.get_variable('convt_w2', [5, 5, 1, n_filter2], initializer=w_init)
        b2 = tf.get_variable('convt_b2', [1], initializer=b_init)
        h2 = tf.sigmoid(tf.nn.conv2d_transpose(h1, w2, output_shape=[tf.shape(h1)[0], 28, 28, 1], strides=[1, 2, 2, 1])+b2)

        y = tf.reshape(h2, [-1, 28*28*1])
    return y

# Gaussian MLP as encoder
def gaussian_MLP_encoder(x, n_hidden, n_output):
    with tf.variable_scope("gaussian_MLP_encoder"):
        # initializers
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.nn.tanh(tf.matmul(x, w0) + b0)

        # output layer-mean
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_output], initializer=w_init)
        b1 = tf.get_variable('b1', [n_output], initializer=b_init)
        mean = tf.matmul(h0, w1) + b1

        # output layer-mean
        w2 = tf.get_variable('w2', [h0.get_shape()[1], n_output], initializer=w_init)
        b2 = tf.get_variable('b2', [n_output], initializer=b_init)
        stddev = tf.matmul(h0, w2) + b2

    return mean, stddev

# Bernoulli MLP as decoder
def bernoulli_MLP_decoder(z, n_hidden, n_output, reuse=False):
    with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.nn.tanh(tf.matmul(z, w0) + b0)

        # output layer-mean
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_output], initializer=w_init)
        b1 = tf.get_variable('b1', [n_output], initializer=b_init)
        y = tf.sigmoid(tf.matmul(h0, w1) + b1)

    return y

# Gateway
def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, TYPE='CNN'):

    if TYPE == 'CNN':
        mu, sigma = gaussian_CNN_encoder(x_hat, dim_z)
        z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        y = bernoulli_CNN_decoder(z)
    else: # MLP
        mu, sigma = gaussian_MLP_encoder(x_hat, n_hidden, dim_z)
        z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        y = bernoulli_MLP_decoder(z, n_hidden, dim_img)

    # loss
    marginal_likelihood = tf.reduce_sum(x * tf.log(1e-8 + y) + (1 - x) * tf.log(1e-8 + 1 - y), 1)
    KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)
    loss = -marginal_likelihood + KL_divergence

    return y, z, loss, -marginal_likelihood, KL_divergence

def decoder(z, dim_img, n_hidden, TYPE='CNN'):
    if TYPE == 'CNN':
        y = bernoulli_CNN_decoder(z, reuse=True)
    else: # MLP
        y = bernoulli_MLP_decoder(z, n_hidden, dim_img, reuse=True)

    return y