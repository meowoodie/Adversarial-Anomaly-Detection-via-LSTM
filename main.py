#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script
"""

import sys
import arrow
import utils
import numpy as np
import scipy.io
import tensorflow as tf
from ppgan import PPGAN
from pprnn import MSTPP_RNN

def generate_fake_test_data(batch_size, seq_len):
    n         = np.random.poisson(lam=int(seq_len/2)) 
    n         = n if n < seq_len else seq_len
    test_t    = np.random.uniform(low=0., high=1., size=[batch_size, n, 1])
    # test_t.sort(axis=1)
    # test_s    = np.random.uniform(low=-1., high=1., size=[batch_size, seq_len, 2])
    # test_t    = np.random.normal(loc=.5, scale=1., size=[batch_size, seq_len, 1])
    test_t.sort(axis=1)
    test_s    = np.random.normal(loc=.0, scale=1., size=[batch_size, n, 2])
    test_data = np.concatenate([test_t, test_s], axis=2)
    return test_data, n

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    with tf.Session() as sess:
        # data preparation
        data       = np.load("data/northcal.earthquake.perseason.npy")
        da         = utils.DataAdapter(init_data=data, S=[[-1., 1.], [-1., 1.]], T=[0., 1.])
        data       = da.normalize(data)[:, 1:101, :]
        mask       = data == 0.
        mask       = mask.astype(float)
        data       = data + mask
        # data = scipy.io.loadmat("data/Tensor1002.mat")["Tensor"][:1000, :, :3]
        print(data.shape)

        # model configurations
        lstm_hidden_size = 10
        # training configurations
        step_size  = np.shape(data)[1]
        batch_size = 20
        test_ratio = 0.3
        epoches    = 50
        lr         = 1e-3
        n_tgrid    = 50
        n_sgrid    = 50

        print(data[0, :, :])

        # TEST MSTPP_RNN
        # 1. define MSTPP_RNN
        pprnn = MSTPP_RNN(step_size, lstm_hidden_size)
        # 2. train via mle
        pprnn.train(sess, batch_size, data, test_ratio, n_tgrid, n_sgrid, epoches, lr)
        # 3. test sequences with different length incrementally
        for seq_len in range(3, 50):
            test_data = data[:batch_size, :seq_len, :]
            test_data = np.concatenate(
                [test_data, np.ones([batch_size, 50 - seq_len, 3])],
                axis=1)
            # # print(test_data)
            # result = pprnn.calculate_log_likelihood_ratio(sess, batch_size, test_data, data[:batch_size, :, :])
            # print(seq_len, result)

        # # TEST PPGAN
        # # 1. define PPGAN
        # ppgan = PPGAN(step_size, lstm_hidden_size, disc_layer_sizes=[20, 10])
        # # 2. train via gan
        # ppgan.train(sess, batch_size, data, test_ratio, epoches, lr)
        # # 3. test sequences with different length incrementally
        # for seq_len in range(2, step_size):
        #     test_fake_data, n = generate_fake_test_data(batch_size, seq_len)
        #     test_real_data    = data[:batch_size, :seq_len, :]
        #     # padding at tail
        #     test_fake_data = np.concatenate(
        #         [test_fake_data, np.ones([batch_size, step_size - n, 3])],
        #         axis=1)
        #     test_real_data = np.concatenate(
        #         [test_real_data, np.ones([batch_size, step_size - seq_len, 3])],
        #         axis=1)
        #     # print(test_data)
        #     fake_result = ppgan.discriminate(sess, batch_size, test_fake_data)
        #     real_result = ppgan.discriminate(sess, batch_size, test_real_data)
        #     print(seq_len, fake_result, real_result)