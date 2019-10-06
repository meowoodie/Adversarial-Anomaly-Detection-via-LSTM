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
from scipy.spatial.distance import cdist

def generate_fake_test_data(batch_size, seq_len):
    n         = np.random.poisson(lam=int(seq_len/2)) 
    n         = n if n < seq_len else seq_len
    test_t    = np.random.uniform(low=0., high=1., size=[batch_size, n, 1])
    test_t.sort(axis=1)
    test_s    = np.random.normal(loc=1.5, scale=1., size=[batch_size, n, 2])
    test_data = np.concatenate([test_t, test_s], axis=2)
    return test_data, n

def calculate_pairwise_distance(real_embeddings, fake_embeddings):
    pw_dist  = cdist(real_embeddings, fake_embeddings, metric="cosine")
    avg_dist = pw_dist.mean(axis=0).mean()
    return avg_dist

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    with tf.Session() as sess:
        # data preparation
        # data = np.load("data/earthquake.npy")[:, :100, :]
        data = scipy.io.loadmat("data/Tensor1005.mat")["Tensor"][:1000, :, :3]
        print(data.shape)

        # model configurations
        lstm_hidden_size = 10
        # training configurations
        step_size  = np.shape(data)[1]

        print(data[0, :, :])

        # # TEST MSTPP_RNN
        # batch_size = 10
        # test_ratio = 0.3
        # epoches    = 5
        # lr         = 1e-1
        # n_tgrid    = 50
        # n_sgrid    = 50
        # # 1. define MSTPP_RNN
        # pprnn = MSTPP_RNN(step_size, lstm_hidden_size)
        # # 2. train via mle
        # pprnn.train(sess, batch_size, data, test_ratio, n_tgrid, n_sgrid, epoches, lr)
        # # 3. test sequences with different length incrementally
        # for seq_len in range(2, step_size):
        #     # shuffle real data order
        #     # shuffled_ids      = np.arange(len(data))
        #     # np.random.shuffle(shuffled_ids)
        #     # data              = data[shuffled_ids]
        #     # prepare test data
        #     test_fake_data, n = generate_fake_test_data(batch_size, seq_len)
        #     test_real_data_0  = data[:batch_size, :, :]
        #     test_real_data_1  = data[batch_size:2*batch_size, :seq_len, :]
        #     # padding at tail
        #     test_fake_data    = np.concatenate(
        #         [test_fake_data, np.ones([batch_size, step_size - n, 3])],
        #         axis=1)
        #     test_real_data_1  = np.concatenate(
        #         [test_real_data_1, np.ones([batch_size, step_size - seq_len, 3])],
        #         axis=1)
        #     # get embeddings for the data
        #     fake_embeddings   = pprnn.get_data_embeddings(sess, test_fake_data)
        #     real_embeddings_0 = pprnn.get_data_embeddings(sess, test_real_data_0)
        #     real_embeddings_1 = pprnn.get_data_embeddings(sess, test_real_data_1)
        #     # return results
        #     fake_result       = calculate_pairwise_distance(real_embeddings_0, fake_embeddings)
        #     real_result       = calculate_pairwise_distance(real_embeddings_0, real_embeddings_1)
        #     print(seq_len, fake_result, real_result)
            

        # TEST PPGAN
        batch_size = 20
        test_ratio = 0.3
        # # for earthquake
        # epoches    = 50
        # lr         = 1e-3
        # for macy's
        epoches    = 15
        lr         = 1e-3
        n_tgrid    = 50
        n_sgrid    = 50
        # 1. define PPGAN
        ppgan = PPGAN(step_size, lstm_hidden_size, disc_layer_sizes=[20, 10])
        # 2. train via gan
        ppgan.train(sess, batch_size, data, test_ratio, epoches, lr)
        # 3. test sequences with different length incrementally
        for seq_len in range(2, step_size):
            test_fake_data, n = generate_fake_test_data(batch_size, seq_len)
            test_real_data    = data[:batch_size, :seq_len, :]
            # padding at tail
            test_fake_data = np.concatenate(
                [test_fake_data, np.ones([batch_size, step_size - n, 3])],
                axis=1)
            test_real_data = np.concatenate(
                [test_real_data, np.ones([batch_size, step_size - seq_len, 3])],
                axis=1)
            # print(test_data)
            fake_result = ppgan.discriminate(sess, batch_size, test_fake_data)
            real_result = ppgan.discriminate(sess, batch_size, test_real_data)
            print(seq_len, fake_result, real_result)