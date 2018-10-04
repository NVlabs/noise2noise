# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import glob
import os
import sys
import argparse
import tensorflow as tf

import PIL.Image
import numpy as np

from collections import defaultdict

size_stats = defaultdict(int)
format_stats = defaultdict(int)

def load_image(fname):
    global format_stats, size_stats
    im = PIL.Image.open(fname)
    format_stats[im.mode] += 1
    if (im.width < 256 or im.height < 256):
        size_stats['< 256x256'] += 1
    else:
        size_stats['>= 256x256'] += 1
    arr = np.array(im.convert('RGB'), dtype=np.uint8)
    assert len(arr.shape) == 3
    return arr.transpose([2, 0, 1])

def shape_feature(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

examples='''examples:

  python %(prog)s --input-dir=./kodak --out=imagenet_val_raw.tfrecords
'''

def main():
    parser = argparse.ArgumentParser(
        description='Convert a set of image files into a TensorFlow tfrecords training set.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", help="Directory containing ImageNet images")
    parser.add_argument("--out", help="Filename of the output tfrecords file")
    args = parser.parse_args()

    if args.input_dir is None:
        print ('Must specify input file directory with --input-dir')
        sys.exit(1)
    if args.out is None:
        print ('Must specify output filename with --out')
        sys.exit(1)

    print ('Loading image list from %s' % args.input_dir)
    images = sorted(glob.glob(os.path.join(args.input_dir, '*.JPEG')))
    images += sorted(glob.glob(os.path.join(args.input_dir, '*.jpg')))
    images += sorted(glob.glob(os.path.join(args.input_dir, '*.png')))
    np.random.RandomState(0x1234f00d).shuffle(images)

    #----------------------------------------------------------
    outdir = os.path.dirname(args.out)
    os.makedirs(outdir, exist_ok=True)
    writer = tf.python_io.TFRecordWriter(args.out)
    for (idx, imgname) in enumerate(images):
        print (idx, imgname)
        image = load_image(imgname)
        feature = {
          'shape': shape_feature(image.shape),
          'data': bytes_feature(tf.compat.as_bytes(image.tostring()))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    print ('Dataset statistics:')
    print ('  Formats:')
    for key in format_stats:
        print ('    %s: %d images' % (key, format_stats[key]))
    print ('  width,height buckets:')
    for key in size_stats:
        print ('    %s: %d images' % (key, size_stats[key]))
    writer.close()



if __name__ == "__main__":
    main()
