#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script defines an adversarial learning framework for MSTPP_RNN defined in `pprnn.py`.

A toy example is also provided at the tail of this script. 
"""

import sys
import arrow
import utils
import numpy as np
import tensorflow as tf

from pprnn import MSTPP_RNN, pack_lstm_states, last_state_before_t

def get_last_encode(mask, encodes):
    """
    helper function for getting the last encode (LSTM hidden state) for each batch
    """
    # mask    [batch_size, step_size]
    # encodes [step_size, batch_size, lstm_hidden_size]

    # size configuration
    b_size = tf.shape(encodes)[1]        # batch_size
    h_size = tf.shape(encodes)[2]        # lstm_hidden_size

    # append a zero state at the begining of the points for each batch
    # NOTE: for t < t_0, a zero state is applied here.
    init_encode = tf.zeros([1, b_size, h_size])
    encodes     = tf.concat([init_encode, encodes], axis=0)

    inds   = tf.reduce_sum(mask, axis=1) # [batch_size]
    i      = tf.range(0, b_size, 1)      # [batch_size]
    last_encode = tf.scan(               # [batch_size, lstm_hidden_size]
        lambda a, x: encodes[x[0], x[1], :],
        (inds, i),
        initializer=tf.zeros(h_size))
    return last_encode

class PPGAN(object):

    def __init__(self, step_size, lstm_hidden_size, disc_layer_sizes):
        """
        """
        self.n_output         = 3
        self.step_size        = step_size
        self.lstm_hidden_size = lstm_hidden_size
        self.disc_layer_sizes = disc_layer_sizes

    def _gan_optimizer(self, batch_size, lr=1e-2):
        """
        adversarial optimizer
        """
        INIT_PARAM_RATIO = 1e-2
        # 1. define generator network with internal sampling that generates fake data 
        with tf.variable_scope("generator") as scope:
            self.generator       = MSTPP_RNN(self.step_size, self.lstm_hidden_size)
            input_fake, _, _, _  = self.generator.create_recurrent_structure(batch_size) # fake input [batch_size, step_size, n_output]
            input_fake           = tf.stack(input_fake, axis=1)
            self.gen_output      = input_fake
        # 2. define discriminator network with external input that takes real data as input
        with tf.variable_scope("discriminator") as scope:
            self.encoder      = MSTPP_RNN(self.step_size, self.lstm_hidden_size)
            self.Ws           = []
            self.bs           = []

            last_layer_size   = self.lstm_hidden_size
            for i in range(len(self.disc_layer_sizes)):
                W = tf.get_variable(name="discW_%d" % i, initializer=INIT_PARAM_RATIO * tf.random_normal([last_layer_size, self.disc_layer_sizes[i]]))
                b = tf.get_variable(name="discb_%d" % i, initializer=INIT_PARAM_RATIO * tf.random_normal([self.disc_layer_sizes[i]]))
                last_layer_size = self.disc_layer_sizes[i]
                self.Ws.append(W)
                self.bs.append(b)
            Wout = tf.get_variable(name="discW_out", initializer=INIT_PARAM_RATIO * tf.random_normal([last_layer_size, 1]))
            bout = tf.get_variable(name="discb_out", initializer=INIT_PARAM_RATIO * tf.random_normal([1]))
            self.Ws.append(Wout)
            self.bs.append(bout)

            # real input [batch_size, step_size, n_output]
            self.input_real   = tf.placeholder(tf.float32, [None, self.step_size, self.n_output]) 
            # encode for real input (step_size [batch_size, lstm_hidden_size])
            _, _, encode_real, mask_real = self.encoder.create_recurrent_structure(batch_size, self.input_real)   
            # encode for fake input (step_size [batch_size, lstm_hidden_size])
            _, _, encode_fake, mask_fake = self.encoder.create_recurrent_structure(batch_size, input_fake)  
            
        encode_real_c, encode_real_h = pack_lstm_states(encode_real)   # 2 * [step_size, batch_size, lstm_hidden_size]
        encode_fake_c, encode_fake_h = pack_lstm_states(encode_fake)   # 2 * [step_size, batch_size, lstm_hidden_size]
        last_encode_real_h = get_last_encode(mask_real, encode_real_h) # [batch_size, lstm_hidden_size]
        last_encode_fake_h = get_last_encode(mask_fake, encode_fake_h) # [batch_size, lstm_hidden_size]
        disc_real = self._discriminator(last_encode_real_h)            # [batch_size, 1]
        disc_fake = self._discriminator(last_encode_fake_h)            # [batch_size, 1]

        self.dr = disc_real
        self.df = disc_fake

        # build Loss
        self.gen_loss   = - tf.reduce_mean(tf.log(disc_fake))
        self.disc_loss  = - tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

        # build optimizers
        gen_vars        = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator") 
        disc_vars       = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        print(gen_vars)
        print(disc_vars)
        optimizer_gen   = tf.train.AdamOptimizer(learning_rate=lr)
        optimizer_disc  = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_gen  = optimizer_gen.minimize(self.gen_loss, var_list=gen_vars)
        self.train_disc = optimizer_disc.minimize(self.disc_loss, var_list=disc_vars)
    
    def _discriminator(self, encode):
        """
        discriminator structure
        """
        # define discriminator weights
        last_layer     = encode # [batch_size, lstm_hidden_size]
        for i in range(len(self.disc_layer_sizes)):
            last_layer = tf.nn.relu(tf.matmul(last_layer, self.Ws[i]) + self.bs[i])
        out_layer      = tf.nn.sigmoid(tf.matmul(last_layer, self.Ws[-1]) + self.bs[-1])
        return out_layer        # [batch_size, 1]

    def discriminate(self, sess, batch_size, data):
        """
        discriminate data samples
        """
        result = sess.run(self.dr, feed_dict={self.input_real: data})
        return result.mean()

    def train(self, sess, batch_size, 
            data,       # external input for the LSTM [n_data, step_size, n_output]
            test_ratio, # fraction of data only for test
            epoches=10, # number of epoches (how many times is the entire dataset going to be trained)
            lr=1e-2):   # learning rate
        """
        training procedure
        """
        # define optimizer
        self._gan_optimizer(batch_size, lr)
        # initialize variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print("[%s] parameters are initialized." % arrow.now(), file=sys.stderr)

        # data configurations
        n_data    = data.shape[0]             # number of data samples
        n_test    = int(n_data * test_ratio)  # number of test samples
        n_train   = n_data - n_test           # number of train samples
        n_batches = int(n_train / batch_size) # number of batches
        # training over epoches
        for epoch in range(epoches):
            # shuffle indices of the training samples
            shuffled_ids = np.arange(n_data)
            np.random.shuffle(shuffled_ids)
            shuffled_train_ids = shuffled_ids[:n_train]
            shuffled_test_ids  = shuffled_ids[-n_test:]

            # training over batches
            avg_train_G_cost = []
            avg_test_G_cost  = []
            avg_train_D_cost = []
            avg_test_D_cost  = []
            avg_dr, avg_df   = [], []
            for b in range(n_batches):
                idx             = np.arange(batch_size * b, batch_size * (b + 1))
                # training and testing indices selected in current batch
                batch_train_ids = shuffled_train_ids[idx]
                batch_test_ids  = shuffled_test_ids[:batch_size]
                # training and testing batch data
                batch_train = data[batch_train_ids, :, :]
                batch_test  = data[batch_test_ids, :, :]
                # optimization procedure
                _, _, train_G_cost, train_D_cost = sess.run(
                    [self.train_gen, self.train_disc, self.gen_loss, self.disc_loss], 
                    feed_dict={self.input_real: batch_train})
                # cost for test batch
                dr, df, test_G_cost, test_D_cost = sess.run(
                    [self.dr, self.df, self.gen_loss, self.disc_loss], 
                    feed_dict={self.input_real: batch_test})
                avg_dr.append(dr)
                avg_df.append(df)
                # # for debug
                # output = sess.run(
                #     [self.gen_output], 
                #     feed_dict={self.input_real: batch_test})
                # print(output)
                # record cost for each batch
                avg_train_G_cost.append(train_G_cost)
                avg_test_G_cost.append(test_G_cost)
                avg_train_D_cost.append(train_D_cost)
                avg_test_D_cost.append(test_D_cost)

            # training log output
            avg_train_G_cost = np.mean(avg_train_G_cost)
            avg_test_G_cost  = np.mean(avg_test_G_cost)
            avg_train_D_cost = np.mean(avg_train_D_cost)
            avg_test_D_cost  = np.mean(avg_test_D_cost)
            avg_dr           = np.concatenate(avg_dr, axis=0).mean()
            avg_df           = np.concatenate(avg_df, axis=0).mean()
            print('[%s] Epoch %d (n_train_batches=%d, batch_size=%d)' % (arrow.now(), epoch, n_batches, batch_size), file=sys.stderr)
            print('[%s] Train cost:\tG:%f\tD:%f' % (arrow.now(), avg_train_G_cost, avg_train_D_cost), file=sys.stderr)
            print('[%s] Test cost:\tG:%f\tD:%f' % (arrow.now(), avg_test_G_cost, avg_test_D_cost), file=sys.stderr)
            print('[%s] Real disc acc:\t%f' % (arrow.now(), avg_dr), file=sys.stderr)
            print('[%s] Fake disc acc:\t%f' % (arrow.now(), avg_df), file=sys.stderr)



# if __name__ == "__main__":
#     np.set_printoptions(suppress=True)
#     # np.random.seed(1)
#     # tf.set_random_seed(1)

#     with tf.Session() as sess:
#         # data preparation
#         data       = np.load("data/northcal.earthquake.perseason.npy")
#         da         = utils.DataAdapter(init_data=data, S=[[-1., 1.], [-1., 1.]], T=[0., 1.])
#         data       = da.normalize(data)[:, 1:51, :]
#         mask       = data == 0.
#         mask       = mask.astype(float)
#         data       = data + mask
#         print(data)
#         # print(data.shape)

#         # model configurations
#         lstm_hidden_size = 10
#         # training configurations
#         step_size  = np.shape(data)[1]
#         batch_size = 5
#         test_ratio = 0.3
#         epoches    = 50
#         lr         = 1e-2
#         n_tgrid    = 50
#         n_sgrid    = 50

#         print(data[0, :, :])

#         # define PPGAN
#         ppgan = PPGAN(step_size, lstm_hidden_size, disc_layer_sizes=[20, 10])

#         # train via gan
#         ppgan.train(sess, batch_size, data, test_ratio, epoches, lr)

#         # test sequences with different length incrementally
#         for seq_len in range(3, 50):
#             test_data = data[:batch_size, :seq_len, :]
#             test_data = np.concatenate(
#                 [test_data, np.ones([batch_size, 50 - seq_len, 3])],
#                 axis=1)
#             # print(test_data)
#             result = ppgan.discriminate(sess, batch_size, test_data)
#             print(seq_len, result)

        