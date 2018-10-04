# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import tensorflow as tf
import numpy as np

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2)):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    std = gain / np.sqrt(fan_in) # He init
    w = tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))
    return w

#----------------------------------------------------------------------------
# Convolutional layer.

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, -1, 1, 1])

def conv2d_bias(x, fmaps, kernel, gain=np.sqrt(2)):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain)
    w = tf.cast(w, x.dtype)
    return apply_bias(tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW'))

def maxpool2d(x, k=2):
    ksize = [1, 1, k, k]
    return tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='SAME', data_format='NCHW')

# TODO use fused upscale+conv2d from gan2
def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

def conv_lr(name, x, fmaps):
    with tf.variable_scope(name):
        return tf.nn.leaky_relu(conv2d_bias(x, fmaps, 3), alpha=0.1)

def conv(name, x, fmaps, gain):
    with tf.variable_scope(name):
        return conv2d_bias(x, fmaps, 3, gain)

def autoencoder(x, width=256, height=256, **_kwargs):
    x.set_shape([None, 3, height, width])

    skips = [x]

    n = x
    n = conv_lr('enc_conv0', n, 48)
    n = conv_lr('enc_conv1', n, 48)
    n = maxpool2d(n)
    skips.append(n)

    n = conv_lr('enc_conv2', n, 48)
    n = maxpool2d(n)
    skips.append(n)

    n = conv_lr('enc_conv3', n, 48)
    n = maxpool2d(n)
    skips.append(n)

    n = conv_lr('enc_conv4', n, 48)
    n = maxpool2d(n)
    skips.append(n)

    n = conv_lr('enc_conv5', n, 48)
    n = maxpool2d(n)
    n = conv_lr('enc_conv6', n, 48)

    #-----------------------------------------------
    n = upscale2d(n)
    n = tf.concat([n, skips.pop()], axis=1)
    n = conv_lr('dec_conv5', n, 96)
    n = conv_lr('dec_conv5b', n, 96)

    n = upscale2d(n)
    n = tf.concat([n, skips.pop()], axis=1)
    n = conv_lr('dec_conv4', n, 96)
    n = conv_lr('dec_conv4b', n, 96)

    n = upscale2d(n)
    n = tf.concat([n, skips.pop()], axis=1)
    n = conv_lr('dec_conv3', n, 96)
    n = conv_lr('dec_conv3b', n, 96)

    n = upscale2d(n)
    n = tf.concat([n, skips.pop()], axis=1)
    n = conv_lr('dec_conv2', n, 96)
    n = conv_lr('dec_conv2b', n, 96)

    n = upscale2d(n)
    n = tf.concat([n, skips.pop()], axis=1)
    n = conv_lr('dec_conv1a', n, 64)
    n = conv_lr('dec_conv1b', n, 32)

    n = conv('dec_conv1', n, 3, gain=1.0)

    return n
