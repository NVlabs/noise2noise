# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

# See README.md in this directory for instructions on how to use this script.

import re
import argparse
import glob
import os
import PIL.Image
import numpy as np
import sys

import util

import nibabel as nib

OUT_RESOLUTION = 256

# Select z-slices from [25,124]
slice_min = 25
slice_max = 125

# Select train and validation subsets from IXI-T1 (these two lists shouldn't overlap)
train_basenames=['IXI002-Guys-0828', 'IXI012-HH-1211', 'IXI013-HH-1212', 'IXI014-HH-1236', 'IXI015-HH-1258', 'IXI016-Guys-0697', 'IXI017-Guys-0698', 'IXI019-Guys-0702', 'IXI020-Guys-0700', 'IXI021-Guys-0703', 'IXI022-Guys-0701', 'IXI023-Guys-0699', 'IXI024-Guys-0705', 'IXI025-Guys-0852', 'IXI026-Guys-0696', 'IXI027-Guys-0710', 'IXI028-Guys-1038', 'IXI029-Guys-0829', 'IXI030-Guys-0708', 'IXI031-Guys-0797', 'IXI033-HH-1259', 'IXI034-HH-1260', 'IXI035-IOP-0873', 'IXI036-Guys-0736', 'IXI037-Guys-0704', 'IXI038-Guys-0729', 'IXI039-HH-1261', 'IXI040-Guys-0724', 'IXI041-Guys-0706', 'IXI042-Guys-0725', 'IXI043-Guys-0714', 'IXI044-Guys-0712', 'IXI045-Guys-0713', 'IXI046-Guys-0824', 'IXI048-HH-1326', 'IXI049-HH-1358', 'IXI050-Guys-0711', 'IXI051-HH-1328', 'IXI052-HH-1343', 'IXI053-Guys-0727', 'IXI054-Guys-0707', 'IXI055-Guys-0730', 'IXI056-HH-1327', 'IXI057-HH-1342', 'IXI058-Guys-0726', 'IXI059-HH-1284', 'IXI060-Guys-0709', 'IXI061-Guys-0715', 'IXI062-Guys-0740', 'IXI063-Guys-0742']
valid_basenames=['IXI064-Guys-0743', 'IXI065-Guys-0744', 'IXI066-Guys-0731', 'IXI067-HH-1356', 'IXI068-Guys-0756', 'IXI069-Guys-0769', 'IXI070-Guys-0767', 'IXI071-Guys-0770', 'IXI072-HH-2324', 'IXI073-Guys-0755']

def fftshift2d(x, ifft=False):
    assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[s0:, :], x[:s0, :]], axis=0)
    x = np.concatenate([x[:, s1:], x[:, :s1]], axis=1)
    return x

def preprocess_mri(input_files,
                   output_file):
    all_files = sorted(input_files)
    num_images = len(all_files)
    print('Input images: %d' % num_images)
    assert num_images > 0

    resolution = np.asarray(PIL.Image.open(all_files[0]), dtype=np.uint8).shape
    assert len(resolution) == 2 # Expect monochromatic images
    print('Image resolution: %s' % str(resolution))

    crop_size = tuple([((r - 1) | 1) for r in resolution])
    crop_slice = np.s_[:crop_size[0], :crop_size[1]]
    print('Crop size: %s' % str(crop_size))

    img_primal = np.zeros((num_images,) + resolution, dtype=np.uint8)
    img_spectrum = np.zeros((num_images,) + crop_size, dtype=np.complex64)

    print('Processing input files..')
    for i, fn in enumerate(all_files):
        if i % 100 == 0:
            print('%d / %d ..' % (i, num_images))
        img = np.asarray(PIL.Image.open(fn), dtype=np.uint8)
        img_primal[i] = img

        img = img.astype(np.float32) / 255.0 - 0.5
        img = img[crop_slice]
        spec = np.fft.fft2(img).astype(np.complex64)
        spec = fftshift2d(spec)
        img_spectrum[i] = spec

    print('Saving: %s' % output_file)
    util.save_pkl((img_primal, img_spectrum), output_file)


def genpng(args):
    if args.outdir is None:
        print ('Must specify output directory with --outdir')
        sys.exit(1)
    if args.ixi_dir is None:
        print ('Must specify input IXI-T1 directory with --ixi-dir')
        sys.exit(1)

    mri_directory = args.ixi_dir

    out_directory = args.outdir
    os.makedirs(out_directory, exist_ok=True)

    nii_files = glob.glob(os.path.join(mri_directory, "*.nii.gz"))

    for nii_file in nii_files:
        print('Processing', nii_file) 
        nii_img = nib.load(nii_file)
        name = os.path.basename(nii_file).split(".")[0]
        print("name", name)
        hborder = (np.asarray([OUT_RESOLUTION, OUT_RESOLUTION]) - nii_img.shape[0:2]) // 2
        print("Img: ", nii_img.shape, " border: ", hborder)
        # Normalize image to [0,1]
        img = nii_img.get_data().astype(np.float32)
        img = img / np.max(img)
        print('Max value', np.max(img))
        # # Slice along z dimension
        #for s in range(70, nii_img.shape[2]-25):
        for s in range(slice_min, slice_max):
            slice = img[:, :, s]
            # Pad to output resolution by inserting zeros
            output = np.zeros([OUT_RESOLUTION, OUT_RESOLUTION])
            output[hborder[0] : hborder[0] + nii_img.shape[0], hborder[1] : hborder[1] + nii_img.shape[1]] = slice
            output = np.minimum(output, 1.0)
            output = np.maximum(output, 0.0)
            output = output * 255

            # Save to png
            if np.max(output) > 1.0:
                outname = os.path.join(out_directory, "%s_%03d.png" % (name, s))
                PIL.Image.fromarray(output).convert('L').save(outname)

def make_slice_name(basename, slice_idx):
    return basename + ('-T1_%03d.png' % slice_idx)

def genpkl(args):
    if args.png_dir is None:
        print ('Must specify PNG directory directory with --png-dir')
        sys.exit(1)
    if args.pkl_dir is None:
        print ('Must specify PKL output directory directory with --pkl-dir')
        sys.exit(1)

    input_train_files = []
    input_valid_files = []
    for base in train_basenames:
        for sidx in range(slice_min, slice_max):
            input_train_files.append(os.path.join(args.png_dir, make_slice_name(base, sidx)))
    for base in valid_basenames:
        for sidx in range(slice_min, slice_max):
            input_valid_files.append(os.path.join(args.png_dir, make_slice_name(base, sidx)))
    print ('Num train samples', len(input_train_files))
    print ('Num valid samples', len(input_valid_files))
    preprocess_mri(input_files=input_train_files, output_file=os.path.join(args.pkl_dir, 'ixi_train.pkl'))
    preprocess_mri(input_files=input_valid_files, output_file=os.path.join(args.pkl_dir, 'ixi_valid.pkl'))

def extract_basenames(lst):
    s = set()
    name_re = re.compile('^(.*)-T1_[0-9]+.png')
    for fname in lst:
        m = name_re.match(os.path.basename(fname))
        if m:
            s.add(m[1])
    return sorted(list(s))

examples='''examples:

  # Convert the IXI-T1 dataset into a set of PNG image files:
  python %(prog)s genpng --ixi-dir=~/Downloads/IXI-T1 --outdir=datasets/ixi-png

  # Convert the PNG image files into a Python pickle for use in training:
  python %(prog)s genpkl --png-dir=datasets/ixi-png --pkl-dir=datasets
'''

def main():
    parser = argparse.ArgumentParser(
        description='Convert the IXI-T1 dataset into a format suitable for network training',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(help='Sub-commands')
    parser_genpng = subparsers.add_parser('genpng', help='IXI nifti to PNG converter (intermediate step)')
    parser_genpng.add_argument('--ixi-dir', help='Directory pointing to unpacked IXI-T1.tar')
    parser_genpng.add_argument('--outdir', help='Directory where to save .PNG files')
    parser_genpng.set_defaults(func=genpng)

    parser_genpkl = subparsers.add_parser('genpkl', help='PNG to PKL converter (used in training)')
    parser_genpkl.add_argument('--png-dir', help='Directory containing .PNGs saved by with the genpng command')
    parser_genpkl.add_argument('--pkl-dir', help='Where to save the .pkl files for train and valid sets')
    parser_genpkl.set_defaults(func=genpkl)

    args = parser.parse_args()
    if 'func' not in args:
        print ('No command given.  Try --help.')
        sys.exit(1)
    args.func(args)

if __name__ == "__main__":
    main()
