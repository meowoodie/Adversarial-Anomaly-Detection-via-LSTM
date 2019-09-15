#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script defines a flexible recurrent neural networks (LSTM) framework for generating 
marked spatio-temporal point processes, as well as couples of helper functions for 
manipulating tensors in tensorflow.

A toy example is also provided at the tail of this script. 
"""

import sys
import arrow
import utils
import numpy as np
import tensorflow as tf

def s_grid(n_int_grid):
    """
    helper function for generating the coordinations of the uniform grids in the spatial region [[-1, 1], [-1, 1]].
    """
    x_bins = np.linspace(-1, 1, n_int_grid)
    y_bins = np.linspace(-1, 1, n_int_grid)
    X, Y   = np.meshgrid(x_bins, y_bins)
    s      = np.concatenate([np.reshape(X, (-1,1)), np.reshape(Y, (-1,1))], axis=-1)
    return s # [n_grid, 3] = [n_int_grid * n_int_grid, 3]

def pack_lstm_states(lstm_states):
    """
    helper function for packing multiple lstm_states into two tensor which keep the information of lstm_states.h
    and lstm_states.c, respectively.
    """
    # lstm_states :                                      (step_size [2, batch_size, lstm_hidden_size])
    C = [ lstm_state.c for lstm_state in lstm_states ] # (step_size [batch_size, lstm_hidden_size])
    H = [ lstm_state.h for lstm_state in lstm_states ] # (step_size [batch_size, lstm_hidden_size])
    return tf.stack(C, axis=0), tf.stack(H, axis=0)    # 2 * [step_size, batch_size, lstm_hidden_size]

def last_state_before_t(t, T, C, H):
    """
    helper function for getting the LSTM states (C and H) of the last moment given the current time t
    """
    # t:    current time                                   scalar
    # T:    time of a batch of points                      [batch_size, step_size]
    # C, H: corresponding LSTM states of a batch of points [step_size, batch_size, lstm_hidden_size]

    # size configuration
    b_size = tf.shape(T)[0] # batch_size
    h_size = tf.shape(H)[2] # lstm_hidden_size

    # append a zero state at the begining of the points for each batch
    # NOTE: for t < t_0, a zero state is applied here.
    init_states = tf.zeros([1, batch_size, lstm_hidden_size])
    C           = tf.concat([init_states, C], axis=0)
    H           = tf.concat([init_states, H], axis=0)

    # retrieve last states given t
    mask = tf.cast(T < t, dtype=tf.int32) # [batch_size, step_size]
    inds = tf.reduce_sum(mask, axis=1)    # [batch_size]
    i    = tf.range(0, b_size, 1)         # [batch_size]
    last_c, last_h = tf.scan(             # 2 * [batch_size, lstm_hidden_size]
        lambda a, x: (
            C[x[0], x[1], :], 
            H[x[0], x[1], :]), 
        (inds, i),
        initializer=(tf.zeros(h_size), tf.zeros(h_size)))
    return last_c, last_h                 # [batch_size, lstm_hidden_size]



class MSTPP_RNN(object):
    """
    Recurrent Neural Networks for Marked Spatio-Temporal Point Processes
    """

    def __init__(self, step_size, lstm_hidden_size):
        """
        Args:
        """
        INIT_PARAM_RATIO = 1 # 1e-2
        # model hyper-parameters
        self.n_output          = 3
        self.lstm_hidden_size  = lstm_hidden_size # size of hidden states
        self.step_size         = step_size        # step size of LSTM
        self.mu                = 0

        # define model weights
        self.W = tf.get_variable(name="W", initializer=INIT_PARAM_RATIO * tf.random_normal([self.lstm_hidden_size, 1]))

    def _recurrent_structure(self, batch_size, is_input=False):
        """Recurrent structure with customized LSTM cells"""
        # define input variable if is_input is True [batch_size, step_size, n_output]
        self.input      = tf.placeholder(tf.float32, [batch_size, self.step_size, self.n_output]) 
        # LSTM structure initialization
        # - create a basic LSTM cell
        self.lstm_cell  = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_size)
        # - define initial basic LSTM hidden state [2, batch_size, lstm_hidden_size]
        #   * lstm_state.h: hidden state [batch_size, lstm_hidden_size]
        #   * lstm_state.c: cell state   [batch_size, lstm_hidden_size]
        init_lstm_state = self.lstm_cell.zero_state(batch_size, dtype=tf.float32)
        # - init_t: initial output [batch_size, 1]
        init_t          = tf.zeros([batch_size], dtype=tf.float32)
        
        outputs = [] # (step_size [batch_size, n_output])
        lams    = [] # (step_size [batch_size, 1])
        states  = [] # (step_size [2, batch_size, lstm_hidden_size])
        last_t, last_lstm_state = init_t, init_lstm_state # loop initialization
        # concatenate each customized LSTM cell iteratively
        for i in range(self.step_size):
            # use external input if is_input is true
            _input = self.input[:, i, :] if is_input is True else None
            # one step in LSTM
            output, lam, lstm_state = self._customized_lstm_cell(batch_size, last_lstm_state, last_t, _input)
            # record outputs and states history
            outputs.append(output)         # [batch_size, n_output]
            lams.append(lam)               # [batch_size, 1]
            states.append(lstm_state)      # [2, batch_size, lstm_hidden_size]
            # update last_t and last_lstm_state
            last_t          = output[:, 0] # [batch_size]
            last_lstm_state = lstm_state   # [2, batch_size, lstm_hidden_size]
        return outputs, lams, states

    def _customized_lstm_cell(self, 
            batch_size, 
            last_lstm_state, # last state as input of this LSTM cell
            last_t,          # get samples from last_t to T
            _input):         # single input
        """
        Customized Stochastic LSTM Cell
        The customized LSTM cell takes external input or generates random samples as input of the next moment. 
        And it returns a single output as well as the hidden state at the next moment.
        """
        if _input is not None:
            # use data as external input to the LSTM
            ts  = _input # [batch_size, n_output]
            lam = self._lambda(ts, last_lstm_state)
        else:
            # sample spatio-temporal points via thinning algorithm
            ts, lam = self._sample_ts(batch_size, last_lstm_state, last_t) # [batch_size, 3]
        # merge spatio-temporal points and marks as final output
        # TODO: add marks
        output = ts
        # one step rnn structure
        # - output is a tensor that contains a single step of data points    [batch_size, n_output]
        # - state contains two tensors in hidden state                       [2, batch_size, lstm_hidden_size]
        _, next_lstm_state = tf.nn.static_rnn(self.lstm_cell, [output], initial_state=last_lstm_state, dtype=tf.float32)
        return output, lam, next_lstm_state

    def _sample_ts(self, batch_size, lstm_state, last_t, n_sample=1000, upperb=1000):
        """
        Sample Single Output (Time and Space)
        Given the last hidden state of the RNN, the function samples a single output (time and space) using 
        thinning algorithm based on the intensity function which is defined by the hidden state. 
        """
        # thinning one spatio-temporal sample for each batch
        ts  = [] # [batch_size, 3]
        lam = [] # [batch_size, 1] 
        for b in range(batch_size):
            # generate random spatio-temporal points in space ([0, 1], [0, 1], [0, 1])
            cand_t   = tf.random.uniform(shape=[n_sample, 1], minval=last_t[b], maxval=1, dtype=tf.float32)
            cand_t   = tf.contrib.framework.sort(cand_t, axis=0) # sort the random points in chronological order
            cand_s   = tf.random.uniform(shape=[n_sample, 2], minval=-1, maxval=1, dtype=tf.float32)
            cand_ts  = tf.concat([cand_t, cand_s], axis=1)
            # generate acceptence rate matrix [batch_size, n_sample]
            accept   = tf.random.uniform(shape=[n_sample, 1], minval=0, maxval=1, dtype=tf.float32)
            # calculate lambda for each sample
            state    = tf.nn.rnn_cell.LSTMStateTuple(
                c=tf.tile(tf.expand_dims(lstm_state.c[b, :], 0), [n_sample, 1]),       # [n_sample, lstm_hidden_size]
                h=tf.tile(tf.expand_dims(lstm_state.h[b, :], 0), [n_sample, 1]))       # [n_sample, lstm_hidden_size]
            cand_lam = self._lambda(cand_ts, state)                                    # [n_sample, 1]
            # reject samples
            mask     = tf.squeeze(tf.cast(accept * upperb > cand_lam, dtype=tf.int32)) # [n_sample]
            # get first non-zero sample
            # NOTE: 
            # the shape of return tensor of tf.gather cannot be inferred, which will lead to a ValueError
            # (Cannot iterate over a shape with unknown rank). Because, function tf.nn.rnn_cell.BasicLSTMCell,
            # requires the shape of inputs should be inferred via shape inference, as shown in
            # https://www.tensorflow.org/api_docs/python/tf/nn/static_rnn
            b_ts  = tf.gather_nd(cand_ts, tf.where(tf.not_equal(mask, 0)))[0]
            b_lam = tf.gather_nd(cand_lam, tf.where(tf.not_equal(mask, 0)))[0]
            ts.append(b_ts)
            lam.append(b_lam)
        ts  = tf.stack(ts)
        lam = tf.stack(lam) 
        return ts, lam

    def _lambda(self, ts, last_lstm_state):
        """
        conditional intensity given history embedding `lstm_state` and current point `ts`
        """
        # ts              [batch_size, 3]
        # last_lstm_state [2, batch_size, lstm_hidden_size]
        # calculate the hidden state for the next moment (the information of current point will be embedded into hidden state)
        _, next_lstm_state = tf.nn.static_rnn(self.lstm_cell, [ts], initial_state=last_lstm_state, dtype=tf.float32)
        # calculate the lambda for the current moment
        lam = tf.exp(tf.linalg.matmul(next_lstm_state.h, self.W))
        return lam # [batch_size]

    def _log_likelihood(self, outputs, lams, states, n_int_grid=10):
        """
        log likelihood given history embedding `lstm_state` and current point `ts`
        """
        # tensors preparation
        # - outputs (step_size [batch_size, n_output])
        # - lams    (step_size [batch_size, 1])
        # - states  (step_size [2, batch_size, lstm_hidden_size])
        outputs = tf.stack(outputs, axis=1) # [batch_size, step_size, 3]
        T       = outputs[:, :, 0]          # [batch_size, step_size]
        C, H    = pack_lstm_states(states)  # [step_size, batch_size, lstm_hidden_size]
        lams    = tf.stack(lams, axis=1)    # [batch_size, step_size, 1]

        # first term: sum of log lambda given all the points 
        loglik_1 = tf.reduce_sum(tf.log(lams), axis=1) # [batch_size, 1]
        
        # second term: integration of lambda over entire spatio-temporal space
        b_size, h_size = tf.shape(outputs)[0], tf.shape(H)[2] # batch_size, lstm_hidden_size
        t    = np.linspace(0, 1, n_int_grid)                  # np: [n_int_grid]
        s    = s_grid(n_int_grid)                             # np: [n_int_grid * n_int_grid, 2]
        c, h = tf.scan(                                       # [n_int_grid, batch_size, lstm_hidden_size]
            lambda a, x: last_state_before_t(x, T, C, H), 
            tf.constant(np.linspace(0, 1, n_int_grid), dtype=tf.float32), 
            initializer=(tf.zeros([batch_size, h_size]), tf.zeros([batch_size, h_size])))
        
        intg = []
        for sj in s:                 # for each spatial point
            for i in range(len(t)):  # for each temporal point
                ts    = tf.constant(np.concatenate([[t[i]], sj]), dtype=tf.float32) # [3]
                ts    = tf.tile(tf.expand_dims(ts, 0), [b_size, 1])                 # [batch_size, 3]
                state = tf.nn.rnn_cell.LSTMStateTuple(                              # [2, batch_size, lstm_hidden_size]
                    c=c[i],                                                         # [batch_size, lstm_hidden_size]
                    h=h[i])                                                         # [batch_size, lstm_hidden_size]
                lam   = self._lambda(ts, state)                                     # [batch_size, 1]
                intg.append(lam)
        loglik_2 = tf.reduce_sum(tf.stack(intg, axis=1), axis=1)                    # [batch_size, 1]

        # third term: sum of log pdf of marks
        # TODO: add marks term

        # calculate log-likelihood
        loglik = loglik_1 + loglik_2
        return loglik # [batch_size, 1]

    def train_gan(self, sess, batch_size):
        """
        """
        # define network structure
        outputs, lams, states = self._recurrent_structure(batch_size)
        loglik = self._log_likelihood(outputs, lams, states)
        # initialize variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        print(sess.run(loglik))

    def train_mle(self, sess, batch_size, 
            data,       # external input for the LSTM [n_data, step_size, n_output]
            test_ratio, # fraction of data only for test
            epoches=10, # number of epoches (how many times is the entire dataset going to be trained)
            lr=1e-2):   # learning rate
        """
        """
        # define network structure
        outputs, lams, states = self._recurrent_structure(batch_size, is_input=True)
        # TODO: add outputs truncations (remove outputs that corresponds to the zero paddings)
        loglik = self._log_likelihood(outputs, lams, states)
        cost   = -loglik
        # Adam optimizer
        global_step   = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps=100, decay_rate=0.99, staircase=True)
        optimizer     = tf.train.AdamOptimizer(learning_rate, beta1=0.6, beta2=0.9).minimize(cost, global_step=global_step)
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
            avg_train_cost = []
            avg_test_cost  = []
            for b in range(n_batches):
                idx             = np.arange(batch_size * b, batch_size * (b + 1))
                # training and testing indices selected in current batch
                batch_train_ids = shuffled_train_ids[idx]
                batch_test_ids  = shuffled_test_ids[:batch_size]
                # training and testing batch data
                batch_train = data[batch_train_ids, :, :]
                batch_test  = data[batch_test_ids, :, :]
                # optimization procedure
                sess.run(optimizer, feed_dict={self.input: batch_train})
                # cost for train batch and test batch
                train_cost = sess.run(cost, feed_dict={self.input: batch_train})
                test_cost  = sess.run(cost, feed_dict={self.input: batch_test})
                print(train_cost, test_cost)
                # record cost for each batch
                avg_train_cost.append(train_cost)
                avg_test_cost.append(test_cost)

            # training log output
            avg_train_cost = np.mean(avg_train_cost)
            avg_test_cost  = np.mean(avg_test_cost)
            print('[%s] Epoch %d (n_train_batches=%d, batch_size=%d)' % (arrow.now(), epoch, n_batches, batch_size), file=sys.stderr)
            print('[%s] Train cost:\t%f' % (arrow.now(), avg_train_cost), file=sys.stderr)
            print('[%s] Test cost:\t%f' % (arrow.now(), avg_test_cost), file=sys.stderr)
        
        # save all training cost into numpy file.
        np.savetxt("train_cost.txt", all_train_cost, delimiter=",")

        

if __name__ == "__main__":
    # np.random.seed(1)
    # tf.set_random_seed(1)

    with tf.Session() as sess:
        # data preparation
        data       = np.load("data/northcal.earthquake.perseason.npy")
        da         = utils.DataAdapter(init_data=data, S=[[-1., 1.], [-1., 1.]], T=[0., 1.])
        data       = da.normalize(data)[:, 1:21, :]
        print(data)
        print(data.shape)
        # model configurations
        lstm_hidden_size = 7
        # training configurations
        step_size  = np.shape(data)[1]
        batch_size = 5 
        test_ratio = 0.1
        epoches    = 10
        lr         = 1e-2
        # define MSTPP_RNN
        pprnn = MSTPP_RNN(step_size, lstm_hidden_size)
        # train via mle
        pprnn.train_mle(sess, batch_size, data, test_ratio, epoches, lr)