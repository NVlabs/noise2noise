# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import numpy as np
import pickle
import PIL.Image

import dnnlib.submission.submit as submit

# save_pkl, load_pkl are used by the mri code to save datasets
def save_pkl(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# save_snapshot, load_snapshot are used save/restore trained networks
def save_snapshot(submit_config, net, fname_postfix):
    dump_fname = os.path.join(submit_config.run_dir, "network_%s.pickle" % fname_postfix)
    with open(dump_fname, "wb") as f:
        pickle.dump(net, f)

def load_snapshot(fname):
    fname = os.path.join(submit.get_path_from_template(fname))
    with open(fname, "rb") as f:
        return pickle.load(f)


def save_image(submit_config, img_t, filename):
    t = img_t.transpose([1, 2, 0])  # [RGB, H, W] -> [H, W, RGB]
    if t.dtype in [np.float32, np.float64]:
        t = clip_to_uint8(t)
    else:
        assert t.dtype == np.uint8
    PIL.Image.fromarray(t, 'RGB').save(os.path.join(submit_config.run_dir, filename))

def clip_to_uint8(arr):
    return np.clip((arr + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)

def crop_np(img, x, y, w, h):
    return img[:, y:h, x:w]

# Run an image through the network (apply reflect padding when needed
# and crop back to original dimensions.)
def infer_image(net, img):
    w = img.shape[2]
    h = img.shape[1]
    pw, ph = (w+31)//32*32-w, (h+31)//32*32-h
    padded_img = img
    if pw!=0 or ph!=0:
        padded_img  = np.pad(img, ((0,0),(0,ph),(0,pw)), 'reflect')
    inferred = net.run(np.expand_dims(padded_img, axis=0), width=w+pw, height=h+ph)
    return clip_to_uint8(crop_np(inferred[0], 0, 0, w, h))
