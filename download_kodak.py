# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import argparse

from urllib.request import urlretrieve

examples='''examples:

  python %(prog)s --output-dir=./tmp
'''

def main():
    parser = argparse.ArgumentParser(
        description='Download the Kodak dataset .PNG image files.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter

    )
    parser.add_argument("--output-dir", help="Directory where to save the Kodak dataset .PNGs")
    args = parser.parse_args()

    if args.output_dir is None:
        print ('Must specify output directory where to store tfrecords with --output-dir')
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(1, 25):
        imgname = 'kodim%02d.png' % i
        url = "http://r0k.us/graphics/kodak/kodak/" + imgname
        print ('Downloading', url)
        urlretrieve(url, os.path.join(args.output_dir, imgname))
    print ('Kodak validation set successfully downloaded.')

if __name__ == "__main__":
    main()
