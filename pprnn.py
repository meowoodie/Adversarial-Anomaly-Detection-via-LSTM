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

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def s_grid(n_sgrid):
    """
    helper function for generating the coordinations of the uniform grids in the spatial region [[-1, 1], [-1, 1]].
    """
    x_bins = np.linspace(-1, 1, n_sgrid)
    y_bins = np.linspace(-1, 1, n_sgrid)
    X, Y   = np.meshgrid(x_bins, y_bins)
    s      = np.concatenate([np.reshape(X, (-1,1)), np.reshape(Y, (-1,1))], axis=-1)
    return tf.constant(s, dtype=tf.float32) # [n_grid, 2] = [n_grid * n_grid, 2]

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
    init_states = tf.zeros([1, b_size, h_size])
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

    def __init__(self, step_size, lstm_hidden_size, external_tensor_input=None):
        """
        Args:
        """
        INIT_PARAM_RATIO       = 1 # 1e-2
        # model hyper-parameters
        self.n_output          = 3
        self.lstm_hidden_size  = lstm_hidden_size # size of hidden states
        self.step_size         = step_size        # step size of LSTM
        self.mu                = 0

        # define model weights
        self.W0 = tf.get_variable(name="W0", initializer=INIT_PARAM_RATIO * tf.random_normal([self.lstm_hidden_size, 1]))
        self.b0 = tf.get_variable(name="b0", initializer=INIT_PARAM_RATIO * tf.random_normal([1]))

        self.W1 = tf.get_variable(name="W1", initializer=INIT_PARAM_RATIO * tf.random_normal([self.lstm_hidden_size, self.n_output]))
        self.b1 = tf.get_variable(name="b1", initializer=INIT_PARAM_RATIO * tf.random_normal([self.n_output]))
        
        # - create a basic LSTM cell
        self.lstm_cell  = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_size)

    def create_recurrent_structure(self, batch_size, lstm_input=None):
        """Recurrent structure with customized LSTM cells"""
        # LSTM structure initialization
        # - define initial basic LSTM hidden state [2, batch_size, lstm_hidden_size]
        #   * lstm_state.h: hidden state [batch_size, lstm_hidden_size]
        #   * lstm_state.c: cell state   [batch_size, lstm_hidden_size]
        init_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(
            c=tf.random_normal([batch_size, self.lstm_hidden_size]), 
            h=tf.random_normal([batch_size, self.lstm_hidden_size])) # self.lstm_cell.zero_state(batch_size, dtype=tf.float32)
        # - init_t: initial output [batch_size, 1]
        init_t          = tf.zeros([batch_size], dtype=tf.float32)
        # - data mask: [batch_size, step_size]
        mask            = tf.cast(lstm_input[:, :, 0] < 1., dtype=tf.int32) if lstm_input is not None else None
        
        outputs = [] # (step_size [batch_size, n_output])
        lams    = [] # (step_size [batch_size, 1])
        states  = [] # (step_size [2, batch_size, lstm_hidden_size])
        last_t, last_lstm_state = init_t, init_lstm_state # loop initialization
        # concatenate each customized LSTM cell iteratively
        for i in range(self.step_size):
            # use external input if is_input is true
            _input = lstm_input[:, i, :] if lstm_input is not None else None
            # one step in LSTM
            output, lam, lstm_state = self._customized_lstm_cell(batch_size, last_lstm_state, last_t, _input)
            # record outputs and states history
            outputs.append(output)         # [batch_size, n_output]
            lams.append(lam)               # [batch_size, 1]
            states.append(lstm_state)      # [2, batch_size, lstm_hidden_size]
            # update last_t and last_lstm_state
            last_t          = output[:, 0] # [batch_size]
            last_lstm_state = lstm_state   # [2, batch_size, lstm_hidden_size]
        return outputs, lams, states, mask

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
        dts = tf.exp(tf.linalg.matmul(lstm_state.h, self.W1) + self.b1) # [batch_size, 3]
        t   = tf.expand_dims(dts[:, 0] + last_t, 1)
        s   = dts[:, 1:]
        ts  = tf.concat([t, s], axis=1)
        lam = self._lambda(ts, lstm_state)
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
        lam = tf.exp(tf.linalg.matmul(next_lstm_state.h, self.W0) + self.b0)
        return lam      # [batch_size]

    def _evaluate_lambda(self, outputs, states, tlim=[0, 1], n_tgrid=15, n_sgrid=15):
        """
        evaluate the lambda value of each point in the specified spatio-temporal space given a sequences of points
        """
        T      = outputs[:, :, 0]                                # [batch_size, step_size]
        C, H   = pack_lstm_states(states)                        # [step_size, batch_size, lstm_hidden_size]
        b_size, h_size = tf.shape(outputs)[0], tf.shape(H)[2]    # batch_size, lstm_hidden_size

        # helper function: replicate LSTM states for n_sgrid * n_sgrid
        def reshape_last_states(x):
            last_c, last_h = last_state_before_t(x, T, C, H)     # [batch_size, lstm_hidden_size]
            last_c = tf.tile(tf.expand_dims(                     # [batch_size, n_sgrid * n_sgrid, lstm_hidden_size]
                last_c, 1), [1, n_sgrid*n_sgrid, 1]) 
            last_c = tf.reshape(                                 # [batch_size * n_sgrid * n_sgrid, lstm_hidden_size]
                last_c, [b_size*n_sgrid*n_sgrid, h_size])
            last_h = tf.tile(tf.expand_dims(                     # [batch_size, n_sgrid * n_sgrid, lstm_hidden_size]
                last_h, 1), [1, n_sgrid*n_sgrid, 1]) 
            last_h = tf.reshape(                                 # [batch_size * n_sgrid * n_sgrid, lstm_hidden_size]
                last_h, [b_size*n_sgrid*n_sgrid, h_size])
            return last_c, last_h
        
        # prepare points (t, s) and states (lstm_states)
        t    = np.linspace(tlim[0], tlim[1], n_tgrid)            # np: [n_tgrid]
        s    = s_grid(n_sgrid)                                   # [n_sgrid * n_sgrid, 2]
        c, h = tf.scan(                                          # [n_tgrid, batch_size * n_sgrid * n_sgrid, lstm_hidden_size]
            lambda a, x: reshape_last_states(x),
            tf.constant(t, dtype=tf.float32), 
            initializer=(
                tf.zeros([b_size*n_sgrid*n_sgrid, h_size]), 
                tf.zeros([b_size*n_sgrid*n_sgrid, h_size])))
        lstm_states = [                                          # (n_tgrid [2, batch_size * n_sgrid * n_sgrid, lstm_hidden_size])
            tf.nn.rnn_cell.LSTMStateTuple(c=c[i], h=h[i]) 
            for i in range(len(t)) ]

        # evaluate lambda for each point
        lam_eval = []                                            # (n_tgrid [batch_size * n_sgrid * n_sgrid, 1])
        for i in range(len(t)):                                  # for each temporal point
            _t  = tf.tile(tf.expand_dims(                        # [n_sgrid * n_sgrid, 1]
                tf.constant([t[i]], dtype=tf.float32), 0),
                [n_sgrid*n_sgrid, 1])
            ts  = tf.concat([_t, s], axis=1)                     # [n_sgrid * n_sgrid, 3]
            ts  = tf.tile(tf.expand_dims(ts, 0), [b_size, 1, 1]) # [batch_size, n_sgrid * n_sgrid, 3]
            ts  = tf.reshape(ts, [b_size*n_sgrid*n_sgrid, 3])    # [batch_size * n_sgrid * n_sgrid, 3]
            lam = self._lambda(ts, lstm_states[i])               # [batch_size * n_sgrid * n_sgrid, 1]
            lam_eval.append(lam)                                 # [n_tgrid, batch_size * n_sgrid * n_sgrid, 1]
        lam_eval = tf.reshape(tf.stack(                          # [batch_size, n_tgrid, n_sgrid, n_sgrid, 1]
            lam_eval, axis=0), [b_size, n_tgrid, n_sgrid, n_sgrid, 1])
        return lam_eval                                          # [batch_size, n_tgrid, n_sgrid, n_sgrid, 1]     

    def log_likelihood(self, outputs, lams, states, mask, n_tgrid, n_sgrid):
        """
        log likelihood given history embedding `lstm_state` and current point `ts`
        """
        # tensors preparation
        # - outputs (step_size [batch_size, n_output])
        # - lams    (step_size [batch_size, 1])
        # - states  (step_size [2, batch_size, lstm_hidden_size])
        outputs  = tf.stack(outputs, axis=1) # [batch_size, step_size, 3]
        lams     = tf.stack(lams, axis=1)    # [batch_size, step_size, 1]

        # first term: sum of log lambda given all the points 
        loglik_1 = tf.squeeze(tf.log(lams))  # [batch_size, step_size]
        loglik_1 = tf.reduce_sum(            # [batch_size, 1]
            tf.multiply(loglik_1, mask), axis=-1) 
        
        # second term: integration of lambda over entire spatio-temporal space
        lam_eval = self._evaluate_lambda(outputs, states, tlim=[0, 1], n_tgrid=n_tgrid, n_sgrid=n_sgrid) 
        loglik_2 = tf.squeeze(tf.reduce_sum(lam_eval, axis=[1, 2, 3])) * \
            tf.constant((1. / n_tgrid) * (2. / n_sgrid) * (2. / n_sgrid), dtype=tf.float32) # [batch_size, 1]

        # third term: sum of log pdf of marks
        # TODO: add marks term

        loglik = loglik_1 - loglik_2
        return loglik # [batch_size]

    def mle_optimizer(self, batch_size, n_tgrid, n_sgrid):
        """
        MLE Optimizer
        """
        # define input variable if external_tensor_input is None [batch_size, step_size, n_output]
        self.lstm_input = tf.placeholder(tf.float32, [None, self.step_size, self.n_output])
        # define network structure with external input
        self.outputs, self.lams, self.states, self.mask = self.create_recurrent_structure(batch_size, self.lstm_input)
        # TODO: add outputs truncations (remove outputs that corresponds to the zero paddings)
        loglik          = self.log_likelihood(self.outputs, self.lams, self.states, self.mask, n_tgrid=n_tgrid, n_sgrid=n_sgrid)
        self.cost       = - tf.reduce_mean(loglik)
        # Adam optimizer
        global_step     = tf.Variable(0, trainable=False)
        learning_rate   = tf.train.exponential_decay(lr, global_step, decay_steps=100, decay_rate=0.99, staircase=True)
        self.optimizer  = tf.train.AdamOptimizer(learning_rate, beta1=0.6, beta2=0.9).minimize(self.cost, global_step=global_step)

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
        self.mle_optimizer(batch_size, n_tgrid, n_sgrid)
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
                sess.run(self.optimizer, feed_dict={self.lstm_input: batch_train})
                # cost for train batch and test batch
                train_cost = sess.run(self.cost, feed_dict={self.lstm_input: batch_train})
                test_cost  = sess.run(self.cost, feed_dict={self.lstm_input: batch_test})
                # for debug
                # outputs, mask = sess.run([tf.stack(self.outputs, axis=1), self.mask], feed_dict={self.lstm_input: batch_train})
                # print(outputs)
                # print(mask)
                # record cost for each batch
                avg_train_cost.append(train_cost)
                avg_test_cost.append(test_cost)

            # training log output
            avg_train_cost = np.mean(avg_train_cost)
            avg_test_cost  = np.mean(avg_test_cost)
            print('[%s] Epoch %d (n_train_batches=%d, batch_size=%d)' % (arrow.now(), epoch, n_batches, batch_size), file=sys.stderr)
            print('[%s] Train cost:\t%f' % (arrow.now(), avg_train_cost), file=sys.stderr)
            print('[%s] Test cost:\t%f' % (arrow.now(), avg_test_cost), file=sys.stderr)

        

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    # np.random.seed(1)
    # tf.set_random_seed(1)

    with tf.Session() as sess:
        # data preparation
        data       = np.load("data/northcal.earthquake.perseason.npy")
        da         = utils.DataAdapter(init_data=data, S=[[-1., 1.], [-1., 1.]], T=[0., 1.])
        data       = da.normalize(data)[:, 1:51, :]
        mask       = data == 0.
        mask       = mask.astype(float)
        data       = data + mask
        print(data)
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

        # define MSTPP_RNN
        pprnn = MSTPP_RNN(step_size, lstm_hidden_size)
        # pprnn.debug(sess)
        # train via mle
        pprnn.train(sess, batch_size, data, test_ratio, n_tgrid, n_sgrid, epoches, lr)
        # pprnn.visualize_lambda(sess, batch_size, data[:20, :, :], tlim=[0, .025], n_tgrid=1000, n_sgrid=20)


    # DEPRECATED: 
    # def visualize_lambda(self, sess, batch_size, data, ind=0, tlim=[0, 1], n_tgrid=1000, n_sgrid=20):
    #     """
    #     Visualize conditional intensity (Lambda) in spatio-temporal space as an animation
    #     given a single trajectory `data` [1, step_size, output_size].
    #     """
    #     print(n_tgrid, n_sgrid)
    #     outputs  = tf.stack(self.outputs, axis=1) # [batch_size, step_size, 3]
    #     lam_eval = self._evaluate_lambda(outputs, self.states, tlim=[0., 1.], n_tgrid=n_tgrid, n_sgrid=n_sgrid) 
    #     lam_eval = tf.squeeze(lam_eval)           # [batch_size, n_tgrid, n_sgrid, n_sgrid]

    #     init_op = tf.global_variables_initializer()
    #     sess.run(init_op)

    #     _lam_eval = sess.run(lam_eval, feed_dict={self.input: data})
    #     print(np.shape(_lam_eval))
    #     utils.plot_spatial_intensity(_lam_eval[0], interval=50)

    # DEPRECATED: NO GRADIENTS FOR TRAINABLE VARIABLES
    # # thinning one spatio-temporal sample for each batch
    # ts  = [] # [batch_size, 3]
    # lam = [] # [batch_size, 1] 
    # for b in range(batch_size):
    #     # generate random spatio-temporal points in space ([0, 1], [0, 1], [0, 1])
    #     cand_t   = tf.random.uniform(shape=[n_sample, 1], minval=last_t[b], maxval=1, dtype=tf.float32)
    #     cand_t   = tf.contrib.framework.sort(cand_t, axis=0) # sort the random points in chronological order
    #     cand_s   = tf.random.uniform(shape=[n_sample, 2], minval=-1, maxval=1, dtype=tf.float32)
    #     cand_ts  = tf.concat([cand_t, cand_s], axis=1)
    #     # generate acceptence rate matrix [batch_size, n_sample]
    #     accept   = tf.random.uniform(shape=[n_sample, 1], minval=0, maxval=1, dtype=tf.float32)
    #     # calculate lambda for each sample
    #     state    = tf.nn.rnn_cell.LSTMStateTuple(
    #         c=tf.tile(tf.expand_dims(lstm_state.c[b, :], 0), [n_sample, 1]),       # [n_sample, lstm_hidden_size]
    #         h=tf.tile(tf.expand_dims(lstm_state.h[b, :], 0), [n_sample, 1]))       # [n_sample, lstm_hidden_size]
    #     cand_lam = self._lambda(cand_ts, state)                                    # [n_sample, 1]
    #     # reject samples
    #     mask     = tf.squeeze(tf.cast(accept * upperb > cand_lam, dtype=tf.int32)) # [n_sample]
    #     # get first non-zero sample
    #     # NOTE: 
    #     # the shape of return tensor of tf.gather cannot be inferred, which will lead to a ValueError
    #     # (Cannot iterate over a shape with unknown rank). Because, function tf.nn.rnn_cell.BasicLSTMCell,
    #     # requires the shape of inputs should be inferred via shape inference, as shown in
    #     # https://www.tensorflow.org/api_docs/python/tf/nn/static_rnn
    #     b_ts  = tf.gather_nd(cand_ts, tf.where(tf.not_equal(mask, 0)))[0]
    #     b_lam = tf.gather_nd(cand_lam, tf.where(tf.not_equal(mask, 0)))[0]
    #     ts.append(b_ts)
    #     lam.append(b_lam)
    # ts  = tf.stack(ts)
    # lam = tf.stack(lam) 