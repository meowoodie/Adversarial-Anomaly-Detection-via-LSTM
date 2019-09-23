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

from pprnn import MSTPP_RNN

class PPGAN(object):

    def __init__(self, step_size, lstm_hidden_size):
        self.n_output         = 3
        self.step_size        = step_size
        self.lstm_hidden_size = lstm_hidden_size
        # 1. define fake generator network with internal sampling that generates fake data 
        with tf.variable_scope("fake") as scope:
            self.fake_generator = MSTPP_RNN(step_size, lstm_hidden_size)
        # 2. define data learner network with external input that takes real data and fake data as input
        with tf.variable_scope("data") as scope:
            # - a. takes real data as input
            self.data_real_learner = MSTPP_RNN(step_size, lstm_hidden_size)

    def gan_optimizer(self, batch_size, n_tgrid, n_sgrid):
        """
        GAN Optimizer
        """
        # 1. create LSTM structure for fake generator
        with tf.variable_scope("fake") as scope:
            self.fake_out, self.fake_lams, self.fake_states = \
                self.fake_generator.create_recurrent_structure(batch_size, is_input=False)
        # 2. create LSTM structure for data learner
        with tf.variable_scope("data") as scope:
            # - a. data learner with real input
            self.data_real_out, self.data_real_lams, self.data_real_states = \
                self.data_real_learner.create_recurrent_structure(batch_size, is_input=True)
            #      the variables will be reused
            scope.reuse_variables() 
            # - b. data learner with fake input
            fake_input             = tf.reshape(tf.stack(self.fake_out, axis=0), [batch_size, self.step_size, self.n_output])
            self.data_fake_learner = MSTPP_RNN(step_size, lstm_hidden_size, external_tensor_input=fake_input)
            self.data_fake_out, self.data_fake_lams, self.data_fake_states = \
                self.data_fake_learner.create_recurrent_structure(batch_size, is_input=True)

        # TODO: add outputs truncations (remove outputs that corresponds to the zero paddings)
        # fake_loglik      = self.fake_generator.log_likelihood(
        #     self.fake_out, self.fake_lams, self.fake_states, n_tgrid=n_tgrid, n_sgrid=n_sgrid) # [batch_size, 1]
        data_real_loglik = self.data_real_learner.log_likelihood(
            self.data_real_out, self.data_real_lams, self.data_real_states, n_tgrid=n_tgrid, n_sgrid=n_sgrid) # [batch_size, 1]
        data_fake_loglik = self.data_fake_learner.log_likelihood(
            self.data_fake_out, self.data_fake_lams, self.data_fake_states, n_tgrid=n_tgrid, n_sgrid=n_sgrid) # [batch_size, 1]

        # improve data learner (discriminator)
        self.D = tf.reduce_sum(data_real_loglik - data_fake_loglik)
        # improve fake generator
        self.G = tf.reduce_sum(data_fake_loglik)

        # Adam optimizer
        generator_variables     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="fake") 
        discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="data")
        global_step      = tf.Variable(0, trainable=False)
        learning_rate    = tf.train.exponential_decay(lr, global_step, decay_steps=100, decay_rate=0.99, staircase=True)
        self.G_optimizer = tf.train.AdamOptimizer(
            learning_rate, beta1=0.6, beta2=0.9).minimize(- self.D, global_step=global_step, var_list=generator_variables)
        self.D_optimizer = tf.train.AdamOptimizer(
            learning_rate, beta1=0.6, beta2=0.9).minimize(- self.G, global_step=global_step, var_list=discriminator_variables)
    
    def train(self, sess, batch_size, 
            data,       # external input for the LSTM [n_data, step_size, n_output]
            test_ratio, # fraction of data only for test
            n_tgrid=20, # number of grid in time 
            n_sgrid=20, # number of grid in space
            epoches=10, # number of epoches (how many times is the entire dataset going to be trained)
            lr=1e-2):   # learning rate
        """
        Training
        """
        # define optimizer
        self.gan_optimizer(batch_size, n_tgrid, n_sgrid)
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
            for b in range(n_batches):
                idx             = np.arange(batch_size * b, batch_size * (b + 1))
                # training and testing indices selected in current batch
                batch_train_ids = shuffled_train_ids[idx]
                batch_test_ids  = shuffled_test_ids[:batch_size]
                # training and testing batch data
                batch_train = data[batch_train_ids, :, :]
                batch_test  = data[batch_test_ids, :, :]
                # optimization procedure
                # sess.run(self.G_optimizer, feed_dict={self.data_real_learner.input: batch_train})
                sess.run(self.D_optimizer, feed_dict={self.data_real_learner.input: batch_train})
                # cost for train batch and test batch
                # train_G_cost = sess.run(self.G, feed_dict={self.data_real_learner.input: batch_train})
                # test_G_cost  = sess.run(self.G, feed_dict={self.data_real_learner.input: batch_test})
                train_D_cost = sess.run(self.D, feed_dict={self.data_real_learner.input: batch_train})
                test_D_cost  = sess.run(self.D, feed_dict={self.data_real_learner.input: batch_test})
                # print(train_cost, test_cost)
                # record cost for each batch
                # avg_train_G_cost.append(train_G_cost)
                # avg_test_G_cost.append(test_G_cost)
                avg_train_D_cost.append(train_D_cost)
                avg_test_D_cost.append(test_D_cost)

            # training log output
            avg_train_G_cost = 0 # np.mean(avg_train_G_cost)
            avg_test_G_cost  = 0 # np.mean(avg_test_G_cost)
            avg_train_D_cost = np.mean(avg_train_D_cost)
            avg_test_D_cost  = np.mean(avg_test_D_cost)
            print('[%s] Epoch %d (n_train_batches=%d, batch_size=%d)' % (arrow.now(), epoch, n_batches, batch_size), file=sys.stderr)
            print('[%s] Train cost:\tG:%f\tD:%f' % (arrow.now(), avg_train_G_cost, avg_train_D_cost), file=sys.stderr)
            print('[%s] Test cost:\tG:%f\tD:%f' % (arrow.now(), avg_test_G_cost, avg_test_D_cost), file=sys.stderr)



if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    # np.random.seed(1)
    # tf.set_random_seed(1)

    with tf.Session() as sess:
        # data preparation
        data       = np.load("data/northcal.earthquake.perseason.npy")
        da         = utils.DataAdapter(init_data=data, S=[[-1., 1.], [-1., 1.]], T=[0., 1.])
        data       = da.normalize(data)[:, 1:21, :]
        # print(data)
        # print(data.shape)

        # model configurations
        lstm_hidden_size = 10
        # training configurations
        step_size  = np.shape(data)[1]
        batch_size = 5
        test_ratio = 0.3
        epoches    = 30
        lr         = 1e-1
        n_tgrid    = 20
        n_sgrid    = 20

        print(data[0, :, :])

        # define PPGAN
        ppgan = PPGAN(step_size, lstm_hidden_size)

        # train via gan
        ppgan.train(sess, batch_size, data, test_ratio, n_tgrid, n_sgrid, epoches, lr)
        # pprnn.visualize_lambda(sess, batch_size, data[:20, :, :], tlim=[0, .025], n_tgrid=1000, n_sgrid=20)