# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import numpy as np
import PIL.Image

import dnnlib.submission.submit as submit
from dnnlib.tflib.autosummary import autosummary

import util

class ValidationSet:
    def __init__(self, submit_config):
        self.images = None
        self.submit_config = submit_config
        return

    def load(self, dataset_dir):
        import glob

        fnames = sorted(glob.glob(os.path.join(submit.get_path_from_template(dataset_dir), '*')))
        images = []
        for fname in fnames:
            try:
                im = PIL.Image.open(fname).convert('RGB')
                arr = np.array(im, dtype=np.float32)
                reshaped = arr.transpose([2, 0, 1]) / 255.0 - 0.5
                images.append(reshaped)
            except OSError as e:
                print ('Skipping file', fname, 'due to error: ', e)
        self.images = images

    def evaluate(self, net, iteration, noise_func):
        avg_psnr = 0.0
        for idx in range(len(self.images)):
            orig_img = self.images[idx]
            w = orig_img.shape[2]
            h = orig_img.shape[1]

            noisy_img = noise_func(orig_img)
            pred255 = util.infer_image(net, noisy_img)
            orig255 = util.clip_to_uint8(orig_img)
            assert (pred255.shape[2] == w and pred255.shape[1] == h)

            sqerr = np.square(orig255.astype(np.float32) - pred255.astype(np.float32))
            s = np.sum(sqerr)
            cur_psnr = 10.0 * np.log10((255*255)/(s / (w*h*3)))
            avg_psnr += cur_psnr

            util.save_image(self.submit_config, pred255, "img_{0}_val_{1}_pred.png".format(iteration, idx))

            if iteration == 0:
                util.save_image(self.submit_config, orig_img, "img_{0}_val_{1}_orig.png".format(iteration, idx))
                util.save_image(self.submit_config, noisy_img, "img_{0}_val_{1}_noisy.png".format(iteration, idx))
        avg_psnr /= len(self.images)
        print ('Average PSNR: %.2f' % autosummary('PSNR_avg_psnr', avg_psnr))
