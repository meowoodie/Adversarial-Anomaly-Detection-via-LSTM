#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script contains all the utilities used in our experiments
"""

import sys
import arrow
import numpy as np

class DataAdapter():
    """
    A helper class for normalizing and restoring data to the specific data range.
    
    init_data: numpy data points with shape [batch_size, seq_len, 3] that defines the x, y, t limits
    S:         data spatial range. eg. [[-1., 1.], [-1., 1.]]
    T:         data temporal range.  eg. [0., 1.]
    """
    def __init__(self, init_data, S=[[-1, 1], [-1, 1]], T=[0., 1.], xlim=None, ylim=None):
        self.data = init_data
        self.T    = T
        self.S    = S
        self.tlim = [ init_data[:, :, 0].min(), init_data[:, :, 0].max() ]
        mask      = np.nonzero(init_data[:, :, 0])
        x_nonzero = init_data[:, :, 1][mask]
        y_nonzero = init_data[:, :, 2][mask]
        self.xlim = [ x_nonzero.min(), x_nonzero.max() ] if xlim is None else xlim
        self.ylim = [ y_nonzero.min(), y_nonzero.max() ] if ylim is None else ylim
        print(self.tlim)
        print(self.xlim)
        print(self.ylim)

    def normalize(self, data):
        """normalize batches of data points to the specified range"""
        rdata = np.copy(data)
        for b in range(len(rdata)):
            # scale x
            rdata[b, np.nonzero(rdata[b, :, 0]), 1] = \
                (rdata[b, np.nonzero(rdata[b, :, 0]), 1] - self.xlim[0]) / \
                (self.xlim[1] - self.xlim[0]) * (self.S[0][1] - self.S[0][0]) + self.S[0][0]
            # scale y
            rdata[b, np.nonzero(rdata[b, :, 0]), 2] = \
                (rdata[b, np.nonzero(rdata[b, :, 0]), 2] - self.ylim[0]) / \
                (self.ylim[1] - self.ylim[0]) * (self.S[1][1] - self.S[1][0]) + self.S[1][0]
            # scale t 
            rdata[b, np.nonzero(rdata[b, :, 0]), 0] = \
                (rdata[b, np.nonzero(rdata[b, :, 0]), 0] - self.tlim[0]) / \
                (self.tlim[1] - self.tlim[0]) * (self.T[1] - self.T[0]) + self.T[0]
        return rdata

    def restore(self, data):
        """restore the normalized batches of data points back to their real ranges."""
        ndata = np.copy(data)
        for b in range(len(ndata)):
            # scale x
            ndata[b, np.nonzero(ndata[b, :, 0]), 1] = \
                (ndata[b, np.nonzero(ndata[b, :, 0]), 1] - self.S[0][0]) / \
                (self.S[0][1] - self.S[0][0]) * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
            # scale y
            ndata[b, np.nonzero(ndata[b, :, 0]), 2] = \
                (ndata[b, np.nonzero(ndata[b, :, 0]), 2] - self.S[1][0]) / \
                (self.S[1][1] - self.S[1][0]) * (self.ylim[1] - self.ylim[0]) + self.ylim[0]
            # scale t 
            ndata[b, np.nonzero(ndata[b, :, 0]), 0] = \
                (ndata[b, np.nonzero(ndata[b, :, 0]), 0] - self.T[0]) / \
                (self.T[1] - self.T[0]) * (self.tlim[1] - self.tlim[0]) + self.tlim[0]
        return ndata

    def normalize_location(self, x, y):
        """normalize a single data location to the specified range"""
        _x = (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (self.S[0][1] - self.S[0][0]) + self.S[0][0]
        _y = (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * (self.S[1][1] - self.S[1][0]) + self.S[1][0]
        return np.array([_x, _y])

    def restore_location(self, x, y):
        """restore a single data location back to the its original range"""
        _x = (x - self.S[0][0]) / (self.S[0][1] - self.S[0][0]) * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
        _y = (y - self.S[1][0]) / (self.S[1][1] - self.S[1][0]) * (self.ylim[1] - self.ylim[0]) + self.ylim[0]
        return np.array([_x, _y])
    
    def __str__(self):
        raw_data_str = "raw data example:\n%s\n" % self.data[:1]
        nor_data_str = "normalized data example:\n%s" % self.normalize(self.data[:1])
        return raw_data_str + nor_data_str