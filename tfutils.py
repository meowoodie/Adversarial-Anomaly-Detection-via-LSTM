#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities in Tensorflow
"""

import tensorflow as tf

def batch_of_vectors_nonzero_entries(batch_of_vectors):
    """
    Removes non-zero entries from batched vectors.

    Requires that each vector have the same number of non-zero entries.

    Args:
    batch_of_vectors: A Tensor with length-N vectors, having shape [..., N].
    Returns:
    A Tensor with shape [..., M] where M is the number of non-zero entries in
    each vector.

    Reference:
    - https://stackoverflow.com/questions/42032517/how-to-omit-zeros-in-a-4-d-tensor-in-tensorflow?rq=1
    """
    nonzero_indices = tf.where(tf.not_equal(batch_of_vectors, tf.zeros_like(batch_of_vectors)))
    # gather_nd gives us a vector containing the non-zero entries of the
    # original Tensor
    nonzero_values = tf.gather_nd(batch_of_vectors, nonzero_indices)
    # Next, reshape so that all but the last dimension is the same as the input
    # Tensor. Note that this will fail unless each vector has the same number of
    # non-zero values.
    reshaped_nonzero_values = tf.reshape(
        nonzero_values,
        tf.concat([tf.shape(batch_of_vectors)[:-1], [-1]], axis=0))
    return reshaped_nonzero_values

t = tf.Variable(
[[[[0., 235., 0., 0., 1006., 0., 0., 23., 42.],
    [77., 0., 0., 12., 0., 0., 33., 55., 0.]],
    [[0., 132., 0., 0., 234., 0., 1., 24., 0.],
    [43., 0., 0., 124., 0., 0., 0., 52., 645]]]])
nonzero_t = batch_of_vectors_nonzero_entries(t)

with tf.Session():
tf.global_variables_initializer().run()
result_evaled = nonzero_t.eval()
print(result_evaled.shape, result_evaled)