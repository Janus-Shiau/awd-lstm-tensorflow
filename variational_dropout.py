'''
Copyright (c) 2019 [Jia-Yau Shiau]
Code work by Jia-Yau (jiayau.shiau@gmail.com).
--------------------------------------------------
Variational Dropout for model regulization implemeted based on:

    https://arxiv.org/abs/1512.05287

"A theoretically Grounded Application of Dropout in Recurrent Neural Networks"
Yarin Gal, Zoubin GhahraMani.
'''
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import math_ops, random_ops
from tensorflow.python.ops.nn_ops import _get_noise_shape

class VariationalDropout(object):
    """ Variational Dropout for model regulization.
        This implemention is based on:
        
            https://arxiv.org/abs/1512.05287

        "A theoretically Grounded Application of Dropout in Recurrent Neural Networks"
        Yarin Gal, Zoubin GhahraMani.
    """
    def __init__(self, input_shape, keep_prob, dtype=None, seed=None):
        self.input_shape = input_shape
        self.keep_prob   = keep_prob
        self.seed        = seed

        self.dtype = tf.float32 if dtype is None else dtype

        self.mask_saved  = tf.get_variable("variational_mask", input_shape, 
            trainable=False, initializer=tf.ones_initializer)
    
    def __call__(self, x):
        """ Apply dropout computation. """
        ret = math_ops.divide(x, self.keep_prob) * self.mask_saved
        
        return ret
    
    def get_update_mask_op(self):
        """ Return a list of the update operation """
        binary_tensor = self._get_binary_mask(self.input_shape, self.dtype)     
        update_op = tf.assign(self.mask_saved, binary_tensor)

        return [update_op]


    def _get_binary_mask(self, noise_shape, dtype):
        """ Compute binary dropout mask. 
            [Inputs]
                noise_shape: the shape of dropout mask.
                dtype: data type of dropout mask.
            [Returns]
                binary_tensor: a dropout mask.
        """
        random_tensor = self.keep_prob
        random_tensor += random_ops.random_uniform(
            noise_shape, seed=self.seed, dtype=dtype)

        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = math_ops.floor(random_tensor)

        return binary_tensor
