# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import math
import time

import numpy as np
import tensorflow as tf
import scipy

import dnnlib
import dnnlib.submission.submit as submit
from dnnlib.tflib.autosummary import autosummary
import dnnlib.tflib.tfutil as tfutil

import config_mri
import util

#----------------------------------------------------------------------------
# The network.

def autoencoder(input):
    def conv(n, name, n_out, size=3, gain=np.sqrt(2)):
        with tf.variable_scope(name):
            wshape = [size, size, int(n.get_shape()[-1]), n_out]
            wstd = gain / np.sqrt(np.prod(wshape[:-1])) # He init
            W = tf.get_variable('W', shape=wshape, initializer=tf.initializers.random_normal(0., wstd))
            b = tf.get_variable('b', shape=[n_out], initializer=tf.initializers.zeros())
            n = tf.nn.conv2d(n, W, strides=[1]*4, padding='SAME')
            n = tf.nn.bias_add(n, b)
        return n

    def up(n, name, f=2):
        with tf.name_scope(name):
            s = [-1 if i.value is None else i.value for i in n.shape]
            n = tf.reshape(n, [s[0], s[1], 1, s[2], 1, s[3]])
            n = tf.tile(n, [1, 1, f, 1, f, 1])
            n = tf.reshape(n, [s[0], s[1] * f, s[2] * f, s[3]])
        return n

    def down(n, name, f=2):     return tf.nn.max_pool(n, ksize=[1, f, f, 1], strides=[1, f, f, 1], padding='SAME', name=name)
    def concat(name, layers):   return tf.concat(layers, axis=-1, name=name)
    def LR(n):                  return tf.nn.leaky_relu(n, alpha=0.1, name='lrelu')

    # Make even size and add the channel dimension.

    input = tf.pad(input, ((0, 0), (0, 1), (0, 1)), 'constant', constant_values=-.5)
    input = tf.expand_dims(input, axis=-1)

    # Encoder part.

    x = input
    x = LR(conv(x, 'enc_conv0', 48))
    x = LR(conv(x, 'enc_conv1', 48))
    x = down(x, 'pool1'); pool1 = x

    x = LR(conv(x, 'enc_conv2', 48))
    x = down(x, 'pool2'); pool2 = x

    x = LR(conv(x, 'enc_conv3', 48))
    x = down(x, 'pool3'); pool3 = x

    x = LR(conv(x, 'enc_conv4', 48))
    x = down(x, 'pool4'); pool4 = x

    x = LR(conv(x, 'enc_conv5', 48))
    x = down(x, 'pool5')

    x = LR(conv(x, 'enc_conv6', 48))

    # Decoder part.

    x = up(x, 'upsample5')
    x = concat('concat5', [x, pool4])
    x = LR(conv(x, 'dec_conv5', 96))
    x = LR(conv(x, 'dec_conv5b', 96))

    x = up(x, 'upsample4')
    x = concat('concat4', [x, pool3])
    x = LR(conv(x, 'dec_conv4', 96))
    x = LR(conv(x, 'dec_conv4b', 96))

    x = up(x, 'upsample3')
    x = concat('concat3', [x, pool2])
    x = LR(conv(x, 'dec_conv3', 96))
    x = LR(conv(x, 'dec_conv3b', 96))

    x = up(x, 'upsample2')
    x = concat('concat2', [x, pool1])
    x = LR(conv(x, 'dec_conv2', 96))
    x = LR(conv(x, 'dec_conv2b', 96))

    x = up(x, 'upsample1')
    x = concat('concat1', [x, input])
    x = LR(conv(x, 'dec_conv1a', 64))
    x = LR(conv(x, 'dec_conv1b', 32))

    x = conv(x, 'dec_conv1', 1, gain=1.0)

    # Remove the channel dimension and crop to odd size.

    return tf.squeeze(x, axis=-1)[:, :-1, :-1]

#----------------------------------------------------------------------------
# Dataset loader

def load_dataset(fn, num_images=None, shuffle=False):
    datadir = submit.get_path_from_template(config_mri.data_dir)
    if fn.lower().endswith('.pkl'):
        abspath = os.path.join(datadir, fn)
        print ('Loading dataset from', abspath)
        img, spec = util.load_pkl(abspath)
    else:
        assert False

    if shuffle:
        perm = np.arange(img.shape[0])
        np.random.shuffle(perm)
        if num_images is not None:
            perm = perm[:num_images]
        img = img[perm]
        spec = spec[perm]

    if num_images is not None:
        img = img[:num_images]
        spec = spec[:num_images]

    # Remove last row/column of the images, we're officially 255x255 now.
    img = img[:, :-1, :-1]

    # Convert to float32.
    assert img.dtype == np.uint8
    img = img.astype(np.float32) / 255.0 - 0.5

    return img, spec

#----------------------------------------------------------------------------
# Dataset iterator.

def fftshift2d(x, ifft=False):
    assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[s0:, :], x[:s0, :]], axis=0)
    x = np.concatenate([x[:, s1:], x[:, :s1]], axis=1)
    return x

bernoulli_mask_cache = dict()
def corrupt_data(img, spec, params):
    ctype = params['type']
    assert ctype == 'bspec'
    p_at_edge = params['p_at_edge']
    global bernoulli_mask_cache
    if bernoulli_mask_cache.get(p_at_edge) is None:
        h = [s // 2 for s in spec.shape]
        r = [np.arange(s, dtype=np.float32) - h for s, h in zip(spec.shape, h)]
        r = [x ** 2 for x in r]
        r = (r[0][:, np.newaxis] + r[1][np.newaxis, :]) ** .5
        m = (p_at_edge ** (1./h[1])) ** r
        bernoulli_mask_cache[p_at_edge] = m
        print('Bernoulli probability at edge = %.5f' % m[h[0], 0])
        print('Average Bernoulli probability = %.5f' % np.mean(m))
    mask = bernoulli_mask_cache[p_at_edge]
    keep = (np.random.uniform(0.0, 1.0, size=spec.shape)**2 < mask)
    keep = keep & keep[::-1, ::-1]
    sval = spec * keep
    smsk = keep.astype(np.float32)
    spec = fftshift2d(sval / (mask + ~keep), ifft=True) # Add 1.0 to not-kept values to prevent div-by-zero.
    img = np.real(np.fft.ifft2(spec)).astype(np.float32)
    return img, sval, smsk

augment_translate_cache = dict()
def augment_data(img, spec, params):
    t = params.get('translate', 0)
    if t > 0:
        global augment_translate_cache
        trans = np.random.randint(-t, t + 1, size=(2,))
        key = (trans[0], trans[1])
        if key not in augment_translate_cache:
            x = np.zeros_like(img)
            x[trans[0], trans[1]] = 1.0
            augment_translate_cache[key] = fftshift2d(np.fft.fft2(x).astype(np.complex64))
        img = np.roll(img, trans, axis=(0, 1))
        spec = spec * augment_translate_cache[key]
    return img, spec

def iterate_minibatches(input_img, input_spec, batch_size, shuffle, corrupt_targets, corrupt_params, augment_params=dict(), max_images=None):
    assert input_img.shape[0] == input_spec.shape[0]

    num = input_img.shape[0]
    all_indices = np.arange(num)
    if shuffle:
        np.random.shuffle(all_indices)

    if max_images:
        all_indices = all_indices[:max_images]
        num = len(all_indices)

    for start_idx in range(0, num, batch_size):
        if start_idx + batch_size <= num:
            indices = all_indices[start_idx : start_idx + batch_size]
            inputs, targets = [], []
            spec_val, spec_mask = [], []

            for i in indices:
                img, spec = augment_data(input_img[i], input_spec[i], augment_params)
                inp, sv, sm = corrupt_data(img, spec, corrupt_params)
                inputs.append(inp)
                spec_val.append(sv)
                spec_mask.append(sm)
                if corrupt_targets:
                    t, _, _ = corrupt_data(img, spec, corrupt_params)
                    targets.append(t)
                else:
                    targets.append(img)

            yield indices, inputs, targets, spec_val, spec_mask

#----------------------------------------------------------------------------
# Training function helpers.

def rampup(epoch, rampup_length):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    return 1.0

def rampdown(epoch, num_epochs, rampdown_length):
    if epoch >= (num_epochs - rampdown_length):
        ep = (epoch - (num_epochs - rampdown_length)) * 0.5
        return math.exp(-(ep * ep) / rampdown_length)
    return 1.0

#----------------------------------------------------------------------------
# Network parameter import/export.

def save_all_variables(fn):
    util.save_pkl({var.name: var.eval() for var in tf.global_variables()}, fn)

#----------------------------------------------------------------------------
# Training function.

def train(submit_config,
          num_epochs            = 300,
          start_epoch           = 0,
          minibatch_size        = 16,
          epoch_train_max       = None,
          post_op               = None,
          corrupt_targets       = True,
          corrupt_params        = dict(),
          augment_params        = dict(),
          rampup_length         = 10,
          rampdown_length       = 30,
          learning_rate_max     = 0.001,
          adam_beta1_initial    = 0.9,
          adam_beta1_rampdown   = 0.5,
          adam_beta2            = 0.99,
          dataset_train         = dict(),
          dataset_test          = dict(),
          load_network          = None):

    result_subdir = submit_config.run_dir

    # Create a run context (hides low level details, exposes simple API to manage the run)
    ctx = dnnlib.RunContext(submit_config, config_mri)
    # Initialize TensorFlow graph and session using good default settings
    tfutil.init_tf(config_mri.tf_config)

    tf.set_random_seed(config_mri.random_seed)
    np.random.seed(config_mri.random_seed)

    print('Loading training set.')
    train_img, train_spec = load_dataset(**dataset_train)

    print('Loading test set.')
    test_img, test_spec = load_dataset(**dataset_test)

    print('Got %d training, %d test images.' % (train_img.shape[0], test_img.shape[0]))
    print('Image size is %d x %d' % train_img.shape[1:])
    print('Spectrum size is %d x %d' % train_spec.shape[1:])

    # Construct TF graph.

    input_shape = [None] + list(train_img.shape)[1:]
    spectrum_shape = [None] + list(train_spec.shape)[1:]
    inputs_var = tf.placeholder(tf.float32, shape=input_shape, name='inputs')
    targets_var = tf.placeholder(tf.float32, shape=input_shape, name='targets')
    denoised = autoencoder(inputs_var)
    denoised_orig = denoised

    if post_op == 'fspec':
        def fftshift3d(x, ifft):
            assert len(x.shape) == 3
            s0 = (x.shape[1] // 2) + (0 if ifft else 1)
            s1 = (x.shape[2] // 2) + (0 if ifft else 1)
            x = tf.concat([x[:, s0:, :], x[:, :s0, :]], axis=1)
            x = tf.concat([x[:, :, s1:], x[:, :, :s1]], axis=2)
            return x

        print('Forcing denoised spectrum to known values.')
        spec_value_var = tf.placeholder(tf.complex64, shape=spectrum_shape, name='spec_value')
        spec_mask_var  = tf.placeholder(tf.float32, shape=spectrum_shape, name='spec_mask')
        denoised_spec  = tf.fft2d(tf.complex(denoised, tf.zeros_like(denoised)))                            # Take FFT of denoiser output.
        denoised_spec  = fftshift3d(denoised_spec, False)                                                   # Shift before applying mask.
        spec_mask_c64  = tf.cast(spec_mask_var, tf.complex64)                                               # TF wants operands to have same type.
        denoised_spec  = spec_value_var * spec_mask_c64 + denoised_spec * (1. - spec_mask_c64)              # Force known frequencies.
        denoised       = tf.cast(tf.spectral.ifft2d(fftshift3d(denoised_spec, True)), dtype = tf.float32)   # Shift back and IFFT.
    else:
        assert post_op is None

    with tf.name_scope('loss'):
        targets_clamped = tf.clip_by_value(targets_var, -0.5, 0.5)
        denoised_clamped = tf.clip_by_value(denoised, -0.5, 0.5)
        # Keep MSE for each item in the minibatch for PSNR computation:
        loss_clamped = tf.reduce_mean((targets_clamped - denoised_clamped)**2, axis=[1,2])
        diff_expr = targets_var - denoised
        loss_test = tf.reduce_mean(diff_expr**2)
        loss_train = loss_test

    # Construct optimization function.

    learning_rate_var = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    adam_beta1_var = tf.placeholder(tf.float32, shape=[], name='adam_beta1')
    train_updates = tf.train.AdamOptimizer(learning_rate=learning_rate_var, beta1=adam_beta1_var, beta2=adam_beta2).minimize(loss_train)

    # Create a log file for Tensorboard
    summary_log = tf.summary.FileWriter(submit_config.run_dir)
    summary_log.add_graph(tf.get_default_graph())

    ctx.update(loss='run %d' % submit_config.run_id, cur_epoch=0, max_epoch=num_epochs)

    # Start training.
    session = tf.get_default_session()
    num_start_epoch = 0 if start_epoch == 'final' else start_epoch
    for epoch in range(num_start_epoch, num_epochs):
        if ctx.should_stop():
            break

        time_start = time.time()

        # Calculate epoch parameters.

        rampup_value = rampup(epoch, rampup_length)
        rampdown_value = rampdown(epoch, num_epochs, rampdown_length)
        adam_beta1 = (rampdown_value * adam_beta1_initial) + ((1.0 - rampdown_value) * adam_beta1_rampdown)
        learning_rate = rampup_value * rampdown_value * learning_rate_max

        # Epoch initializer.

        if epoch == num_start_epoch:
            session.run(tf.global_variables_initializer(), feed_dict={adam_beta1_var: adam_beta1})
            if load_network:
                print('Loading network %s' % load_network)
                var_dict = util.load_pkl(os.path.join(result_subdir, load_network))
                for var in tf.global_variables():
                    if var.name in var_dict:
                        tf.assign(var, var_dict[var.name]).eval()

        # Skip the rest if we only want the final inference.

        if start_epoch == 'final':
            break

        # Train.

        train_loss, train_n = 0., 0.
        for (indices, inputs, targets, input_spec_val, input_spec_mask) in iterate_minibatches(train_img, train_spec, batch_size=minibatch_size, shuffle=True, corrupt_targets=corrupt_targets, corrupt_params=corrupt_params, augment_params=augment_params, max_images=epoch_train_max):
            # Export example training pairs from the first batch.
            if epoch == num_start_epoch and train_n == 0:
                img = np.concatenate([np.concatenate(inputs[:6], axis=1), np.concatenate(targets[:6], axis=1)], axis=0) + 0.5
                scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(os.path.join(result_subdir, 'example_train.png'))

            # Construct feed dictionary.
            feed_dict = {inputs_var: inputs, targets_var: targets, learning_rate_var: learning_rate, adam_beta1_var: adam_beta1}
            if post_op == 'fspec':
                feed_dict.update({spec_value_var: input_spec_val, spec_mask_var: input_spec_mask})

            # Run.
            loss_val, _ = session.run([loss_train, train_updates], feed_dict=feed_dict)

            # Stats.
            train_loss += loss_val * len(indices)
            train_n += len(indices)

        train_loss /= train_n

        # Test.

        test_db_clamped = 0.0
        test_n = 0.0
        for (indices, inputs, targets, input_spec_val, input_spec_mask) in iterate_minibatches(test_img, test_spec, batch_size=minibatch_size, shuffle=False, corrupt_targets=False, corrupt_params=corrupt_params):
            # Construct feed dictionary.
            feed_dict = {inputs_var: inputs, targets_var: targets}
            if post_op == 'fspec':
                feed_dict.update({spec_value_var: input_spec_val, spec_mask_var: input_spec_mask})

            # Run.
            if test_n == 0:
                # Export example result.
                loss_clamped_vals, orig, outputs = session.run([loss_clamped, denoised_orig, denoised], feed_dict=feed_dict)
                prim = [inputs[0], orig[0], outputs[0], targets[0]]
                spec = [fftshift2d(abs(np.fft.fft2(x))) for x in prim]
                pimg = np.concatenate(prim, axis=1) + 0.5
                simg = np.concatenate(spec, axis=1) * 0.03
                img = np.concatenate([pimg, simg], axis=0)
                scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(os.path.join(result_subdir, 'img%05d.png' % epoch))
            else:
                # Just run.
                loss_clamped_vals, = session.run([loss_clamped], feed_dict=feed_dict)

            # Stats.
            indiv_db = 10 * np.log10(1.0 / loss_clamped_vals)
            test_db_clamped += np.sum(indiv_db)
            test_n += len(indices)

        test_db_clamped /= test_n

        # Export network.

        if epoch % 10 == 0:
            save_all_variables(os.path.join(result_subdir, 'network-snapshot-%05d.pkl' % epoch))

        # Export and print stats, update progress monitor.

        time_epoch = time.time() - time_start

        print('Epoch %3d/%d: time=%7.3f, train_loss=%.7f, test_db_clamped=%.5f, lr=%7f' % (
            epoch, num_epochs,
            autosummary("Timing/sec_per_epoch", time_epoch),
            autosummary("Train/loss", train_loss),
            autosummary("Test/dB_clamped", test_db_clamped),
            autosummary("Learning_rate", learning_rate)
        ))
        ctx.update(loss='run %d' % submit_config.run_id, cur_epoch=epoch, max_epoch=num_epochs)
        dnnlib.tflib.autosummary.save_summaries(summary_log, epoch)

    # Training done, session still around. Save final network weights.
    print("Saving final network weights.")
    save_all_variables(os.path.join(result_subdir, 'network-final.pkl'))

    print("Resetting random seed and saving a bunch of example images.")
    np.random.seed(config_mri.random_seed)
    idx = 0
    with open(os.path.join(result_subdir, 'psnr.txt'), 'wt') as fout:
        for (indices, inputs, targets, input_spec_val, input_spec_mask) in iterate_minibatches(test_img, test_spec, batch_size=1, shuffle=True, corrupt_targets=False, corrupt_params=corrupt_params):
            # Construct feed dictionary.
            feed_dict = {inputs_var: inputs, targets_var: targets}
            if post_op == 'fspec':
                feed_dict.update({spec_value_var: input_spec_val, spec_mask_var: input_spec_mask})

            # Export example result.
            loss_val, loss_clamped_val, orig, outputs = session.run([loss_test, loss_clamped, denoised_orig, denoised], feed_dict=feed_dict)
            prim = [inputs[0], orig[0], outputs[0], targets[0]]
            spec = [fftshift2d(abs(np.fft.fft2(x))) for x in prim]
            pimg = np.concatenate(prim, axis=1) + 0.5
            simg = np.concatenate(spec, axis=1) * 0.03
            img = np.concatenate([pimg, simg], axis=0)
            scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(os.path.join(result_subdir, 'final%05d.png' % idx))

            input_clamped = np.clip(inputs[0], -.5, .5)
            trg_clamped = np.clip(targets[0], -.5, .5)
            db = -10.0 * math.log10(loss_val)
            db_clamped = -10.0 * math.log10(loss_clamped_val)
            db_input = -10.0 * math.log10(np.mean((input_clamped - trg_clamped)**2))

            fout.write('%3d: %.5f %.5f %.5f\n' % (idx, db_input, db, db_clamped))

            idx += 1
            if idx == 100:
                break

    # Session deleted, clean up.
    summary_log.close()
    ctx.close()
