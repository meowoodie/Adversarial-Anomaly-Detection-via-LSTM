#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Recurrent Neural Networks for Point Processes.
"""

import sys
import arrow
import numpy as np
import tensorflow as tf

class PointProcessRNN(object):
    """
    """

    def __init__(self, step_size, lstm_hidden_size, ts_len):
        """
        Params:
        """
        INIT_PARAM_RATIO = 1e-2
        # model hyper-parameters
        self.lstm_hidden_size  = lstm_hidden_size # size of hidden states
        self.step_size         = step_size        # step size of LSTM
        self.ts_len            = ts_len           # length of a single output
        self.mu                = 0

        # # define model weights
        self.K = tf.get_variable(name="K", initializer=INIT_PARAM_RATIO * tf.random_normal([3, self.lstm_hidden_size]))
        # # - time weights
        # self.Wt  = tf.get_variable(name="Wt", initializer=INIT_PARAM_RATIO * tf.random_normal([self.lstm_hidden_size, self.t_dim]))
        # self.bt  = tf.get_variable(name="bt", initializer=INIT_PARAM_RATIO * tf.random_normal([self.t_dim]))
        # # - location weights
        # self.Wl0 = tf.get_variable(name="Wl0", initializer=INIT_PARAM_RATIO * tf.random_normal([self.lstm_hidden_size, self.loc_hidden_size]))
        # self.bl0 = tf.get_variable(name="bl0", initializer=INIT_PARAM_RATIO * tf.random_normal([self.loc_hidden_size]))
        # self.Wl1 = tf.get_variable(name="Wl1", initializer=INIT_PARAM_RATIO * tf.random_normal([self.loc_hidden_size, self.loc_param_size]))
        # self.bl1 = tf.get_variable(name="bl1", initializer=INIT_PARAM_RATIO * tf.random_normal([self.loc_param_size]))
        # # - mark weights
        # self.Wm0 = tf.get_variable(name="Wm0", initializer=INIT_PARAM_RATIO * tf.random_normal([self.lstm_hidden_size, self.mak_hidden_size]))
        # self.bm0 = tf.get_variable(name="bm0", initializer=INIT_PARAM_RATIO * tf.random_normal([self.mak_hidden_size]))
        # self.Wm1 = tf.get_variable(name="Wm1", initializer=INIT_PARAM_RATIO * tf.random_normal([self.mak_hidden_size, self.m_dim]))
        # self.bm1 = tf.get_variable(name="bm1", initializer=INIT_PARAM_RATIO * tf.random_normal([self.m_dim]))

    def _recurrent_structure(self, batch_size):
        """Recurrent structure with customized LSTM cells."""
        # create a basic LSTM cell
        tf_lstm_cell    = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_size)
        # defining initial basic LSTM hidden state [2, batch_size, lstm_hidden_size]
        # - lstm_state.h: hidden state [batch_size, lstm_hidden_size]
        # - lstm_state.c: cell state   [batch_size, lstm_hidden_size]
        init_lstm_state = tf_lstm_cell.zero_state(batch_size, dtype=tf.float32)
        # init_ts: initial output [batch_size, ts_len]
        init_ts         = tf.zeros([batch_size, self.ts_len], dtype=tf.float32)
        # concatenate each customized LSTM cell by loop
        output = []
        loglik = []
        last_ts, last_lstm_state = init_ts, init_lstm_state # loop initialization
        for _ in range(self.step_size):
            ts, l, state = self._customized_lstm_cell(batch_size, tf_lstm_cell, last_lstm_state, last_ts)
            output.append(ts)       # record single output
            loglik.append(l)        # record likelihood
            last_ts         = ts    # reset last_ts
            last_lstm_state = state # reset last_lstm_state
        output = tf.stack(output, axis=1) # [batch_size, step_size, ts_len]
        loglik = tf.stack(loglik, axis=1) # [batch_size, step_size, 1]
        return output, loglik, state

    def _customized_lstm_cell(self, 
            batch_size, 
            tf_lstm_cell,    # tensorflow LSTM cell object, e.g. 'tf.nn.rnn_cell.BasicLSTMCell'
            last_lstm_state, # last state as input of this LSTM cell
            last_t):         # last_t + delta_t as input of this LSTM cell
        """
        Customized Stochastic LSTM Cell
        The customized LSTM cell takes external input and the hidden state of the last moment
        as input, then return the external output as well as the hidden state at the next moment. 
        The reason that avoids using tensorflow builtin rnn structure is that, 
        """
        # stochastic single output and its likelihood
        ts, l = self._sample_ts(batch_size, last_lstm_state.h) # [batch_size, ts_len], [batch_size, 1]
        # one step rnn structure
        # - ts is a tensor that contains a single step of data points with shape [batch_size, ts_len]
        # - state is a tensor of hidden state with shape [2, batch_size, state_size]
        _, next_state = tf.nn.static_rnn(tf_lstm_cell, [ts], initial_state=last_lstm_state, dtype=tf.float32)
        return ts, l, next_state

    def _sample_ts(self, batch_size, lstm_state, n_sample=1000, upperb=1000):
        """
        Sample Single Output and Its Likelihood Value
        Given the last hidden state of the RNN, the function samples a single output based on 
        the intensity function which is defined by the hidden state. 
        """
        # generate random spatio-temporal points in space ([0, 1], [0, 1], [0, 1])
        cand_t  = tf.random.uniform(shape=[batch_size, n_sample, 1], minval=0, maxval=1, dtype=tf.float32)
        cand_t  = tf.contrib.framework.sort(cand_t, axis=1) # sort the random points in chronological order
        cand_s  = tf.random.uniform(shape=[batch_size, n_sample, 2], minval=-1, maxval=1, dtype=tf.float32)
        cand_ts = tf.concat([cand_t, cand_s], axis=2)
        # generate acceptence rate matrix [batch_size, n_sample]
        accept  = tf.random.uniform(shape=[batch_size, n_sample, 1], minval=0, maxval=1, dtype=tf.float32)
        # calculate lambda (conditional intensity) for each batch
        ts      = []
        for b in range(batch_size):
            b_ts_cand    = cand_ts[b, :, :]                                               # [n_sample, 3]
            b_lstm_state = tf.expand_dims(lstm_state[b, :], -1)                           # [lstm_hidden_size]
            b_accept     = accept[b, :, :]                                                # [n_sample, 1]
            b_lam        = self.mu + tf.exp(tf.linalg.matmul(tf.linalg.matmul(b_ts_cand, self.K), b_lstm_state)) # [n_sample, 1]
            # reject samples
            b_mask       = tf.squeeze(tf.cast(b_accept * upperb > b_lam, dtype=tf.int32)) # [n_sample]
            # get first non-zero sample
            b_ts         = tf.gather(
                b_ts_cand,
                tf.squeeze(tf.where(tf.not_equal(b_mask, 0)))[0])
            ts.append(b_ts)
        ts = tf.stack(ts)
        return ts, 0

    def train(self, sess, batch_size):
        """
        """
        # define network structure
        self.output, self.loglik, self.final_state = self._recurrent_structure(batch_size)
        # initialize variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 
        print(sess.run(self.output))

    def debug(self, sess, batch_size):
        tf_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_size)
        lstm_state   = tf_lstm_cell.zero_state(5, dtype=tf.float32).h
        ts, l = self._sample_ts(batch_size, lstm_state=lstm_state, n_sample=10, upperb=2)
        _, next_state = tf.nn.static_rnn(tf_lstm_cell, [ts], initial_state=lstm_state, dtype=tf.float32)

        # initialize variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        print(sess.run(next_state))
        



if __name__ == "__main__":
    np.random.seed(1)
    tf.set_random_seed(1)

    data = np.random.rand(10, 10, 1)
    data.sort(axis=1)
    # print(data)

    batch_size       = 5
    step_size        = 10
    lstm_hidden_size = 7
    ts_len           = 3

    with tf.Session() as sess:
        
        pprnn = PointProcessRNN(step_size, lstm_hidden_size, ts_len)
        
        pprnn.debug(sess, batch_size)