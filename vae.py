import tensorflow as tf
from tensorbayes.layers import dense, placeholder
from tensorbayes.utils import progbar
from tensorbayes.tfutils import binary_crossentropy
import numpy as np

class vanilla_vae:
    """
    build a vanilla vae
    you can customize the activation functions pf each layer yourself.
    """

    def __init__(self, input_dim, encoding_dims, z_dim, decoding_dims, loss="cross_entropy", useTranse = False, eps = 1e-10):
        # useTranse: if we use trasposed weigths of inference nets
        # eps for numerical stability
        # structural info
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.encoding_dims = encoding_dims
        self.decoding_dims = decoding_dims
        self.loss = loss
        self.useTranse = useTranse
        self.eps = eps
        self.weights = []    # better in np form. first run, then append in
        self.bias = []



    def fit(self, x_input, epochs = 1000, learning_rate = 0.001, batch_size = 100, print_size = 50, train=True):
        # training setting
        self.DO_SHARE = False
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.print_size = print_size

        self.g = tf.Graph()
        # inference process
        x_ = placeholder((None, self.input_dim))
        x = x_
        depth_inf = len(self.encoding_dims)
        for i in range(depth_inf):
            x = dense(x, self.encoding_dims[i], scope="enc_layer"+"%s" %i, activation=tf.nn.sigmoid)
        h_encode = x
        z_mu = dense(h_encode, self.z_dim, scope="mu_layer")
        z_log_sigma_sq = dense(h_encode, self.z_dim, scope = "sigma_layer")
        e = tf.random_normal(tf.shape(z_mu))
        z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_log_sigma_sq), self.eps)) * e

        # generative process
        if self.useTranse == False:
            depth_gen = len(self.decoding_dims)

            for i in range(depth_gen):
                y = dense(z, self.decoding_dims[i], scope="dec_layer"+"%s" %i, activation=tf.nn.sigmoid)
                # if last_layer_nonelinear: depth_gen -1

        else:
            depth_gen = depth_inf
            ## haven't finnished yet...

        x_recons = y

        if self.loss == "relu":
            loss_recons = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(x_recons, x_, self.eps), axis=1))
            loss_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_mu) + tf.exp(z_log_sigma_sq) - z_log_sigma_sq - 1, 1))
            # loss_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_mu) + tf.exp(z_log_sigma_sq) - z_log_sigma_sq - 1, 1))
            loss = loss_recons + loss_kl
        # other cases not finished yet
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt_dir = "pre_model/" + "vae.ckpt"
        if train == True:
            # num_turn = x_input.shape[0] / self.batch_size
            for i in range(epochs):
                idx = np.random.choice(x_input.shape[0], batch_size, replace=False)
                x_batch = x_input[idx]
                _, l = sess.run((train_op, loss), feed_dict={x_:x_batch})
                if i % self.print_size == 0:
                    print "{:>10s}{:>10s}".format("epoces","loss")
                    print "{:10.2e}{:10.2e}".format(i, l)
            saver.save(sess, ckpt_dir)
        else:
            saver.restore(sess, ckpt_dir)
