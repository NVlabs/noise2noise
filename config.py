# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import dnnlib
import argparse
import sys

import dnnlib.submission.submit as submit

import validation

# Submit config
# ------------------------------------------------------------------------------------------

submit_config = dnnlib.SubmitConfig()
submit_config.run_dir_root = 'results'
submit_config.run_dir_ignore += ['datasets', 'results']

desc = "autoencoder"

# Tensorflow config
# ------------------------------------------------------------------------------------------

tf_config = dnnlib.EasyDict()
tf_config["graph_options.place_pruned_graph"] = True

# Network config
# ------------------------------------------------------------------------------------------

net_config = dnnlib.EasyDict(func_name="network.autoencoder")

# Optimizer config
# ------------------------------------------------------------------------------------------

optimizer_config = dnnlib.EasyDict(beta1=0.9, beta2=0.99, epsilon=1e-8)

# Noise augmentation config
gaussian_noise_config = dnnlib.EasyDict(
    func_name='train.AugmentGaussian',
    train_stddev_rng_range=(0.0, 50.0),
    validation_stddev=25.0
)
poisson_noise_config = dnnlib.EasyDict(
    func_name='train.AugmentPoisson',
    lam_max=50.0
)

# ------------------------------------------------------------------------------------------
# Preconfigured validation sets
datasets = {
    'kodak':  dnnlib.EasyDict(dataset_dir='datasets/kodak'),
    'bsd300': dnnlib.EasyDict(dataset_dir='datasets/bsd300'),
    'set14':  dnnlib.EasyDict(dataset_dir='datasets/set14')
}

default_validation_config = datasets['kodak']

corruption_types = {
    'gaussian': gaussian_noise_config,
    'poisson': poisson_noise_config
}

# Train config
# ------------------------------------------------------------------------------------------

train_config = dnnlib.EasyDict(
    iteration_count=300000,
    eval_interval=1000,
    minibatch_size=4,
    run_func_name="train.train",
    learning_rate=0.0003,
    ramp_down_perc=0.3,
    noise=gaussian_noise_config,
#    noise=poisson_noise_config,
    noise2noise=True,
    train_tfrecords='datasets/imagenet_val_raw.tfrecords',
    validation_config=default_validation_config
)

# Validation run config
# ------------------------------------------------------------------------------------------
validate_config = dnnlib.EasyDict(
    run_func_name="validation.validate",
    dataset=default_validation_config,
    network_snapshot=None,
    noise=gaussian_noise_config
)

# ------------------------------------------------------------------------------------------

# jhellsten quota group

def error(*print_args):
    print (*print_args)
    sys.exit(1)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ------------------------------------------------------------------------------------------
examples='''examples:

  # Train a network using the BSD300 dataset:
  python %(prog)s train --train-tfrecords=datasets/bsd300.tfrecords

  # Run a set of images through a pre-trained network:
  python %(prog)s validate --network-snapshot=results/network_final.pickle --dataset-dir=datasets/kodak
'''

if __name__ == "__main__":
    def train(args):
        if args:
            n2n = args.noise2noise if 'noise2noise' in args else True
            train_config.noise2noise = n2n
            if 'long_train' in args and args.long_train:
                train_config.iteration_count = 500000
                train_config.eval_interval = 5000
                train_config.ramp_down_perc = 0.5
        else:
            print ('running with defaults in train_config')
        noise = 'gaussian'
        if 'noise' in args:
            if args.noise not in corruption_types:
                error('Unknown noise type', args.noise)
            else:
                noise = args.noise
        train_config.noise = corruption_types[noise]

        if train_config.noise2noise:
            submit_config.run_desc += "-n2n"
        else:
            submit_config.run_desc += "-n2c"

        if 'train_tfrecords' in args and args.train_tfrecords is not None:
            train_config.train_tfrecords = submit.get_path_from_template(args.train_tfrecords)

        print (train_config)
        dnnlib.submission.submit.submit_run(submit_config, **train_config)

    def validate(args):
        if submit_config.submit_target != dnnlib.SubmitTarget.LOCAL:
            print ('Command line overrides currently supported only in local runs for the validate subcommand')
            sys.exit(1)
        if args.dataset_dir is None:
            error('Must select dataset with --dataset-dir')
        else:
            validate_config.dataset = {
                'dataset_dir': args.dataset_dir
            }
        if args.noise not in corruption_types:
            error('Unknown noise type', args.noise)
        validate_config.noise = corruption_types[args.noise]
        if args.network_snapshot is None:
            error('Must specify trained network filename with --network-snapshot')
        validate_config.network_snapshot = args.network_snapshot
        dnnlib.submission.submit.submit_run(submit_config, **validate_config)

    def infer_image(args):
        if submit_config.submit_target != dnnlib.SubmitTarget.LOCAL:
            print ('Command line overrides currently supported only in local runs for the validate subcommand')
            sys.exit(1)
        if args.image is None:
            error('Must specify image file with --image')
        if args.out is None:
            error('Must specify output image file with --out')
        if args.network_snapshot is None:
            error('Must specify trained network filename with --network-snapshot')
        # Note: there's no dnnlib.submission.submit_run here. This is for quick interactive
        # testing, not for long-running training or validation runs.
        validation.infer_image(args.network_snapshot, args.image, args.out)

    # Train by default
    parser = argparse.ArgumentParser(
        description='Train a network or run a set of images through a trained network.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--desc', default='', help='Append desc to the run descriptor string')
    parser.add_argument('--run-dir-root', help='Working dir for a training or a validation run. Will contain training and validation results.')
    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')
    parser_train = subparsers.add_parser('train', help='Train a network')
    parser_train.add_argument('--noise2noise', nargs='?', type=str2bool, const=True, default=True, help='Noise2noise (--noise2noise=true) or noise2clean (--noise2noise=false).  Default is noise2noise=true.')
    parser_train.add_argument('--noise', default='gaussian', help='Type of noise corruption (one of: gaussian, poisson)')
    parser_train.add_argument('--long-train', default=False, help='Train for a very long time (500k iterations or 500k*minibatch image)')
    parser_train.add_argument('--train-tfrecords', help='Filename of the training set tfrecords file')
    parser_train.set_defaults(func=train)

    parser_validate = subparsers.add_parser('validate', help='Run a set of images through the network')
    parser_validate.add_argument('--dataset-dir', help='Load all images from a directory (*.png, *.jpg/jpeg, *.bmp)')
    parser_validate.add_argument('--network-snapshot', help='Trained network pickle')
    parser_validate.add_argument('--noise', default='gaussian', help='Type of noise corruption (one of: gaussian, poisson)')
    parser_validate.set_defaults(func=validate)

    parser_infer_image = subparsers.add_parser('infer-image', help='Run one image through the network without adding any noise')
    parser_infer_image.add_argument('--image', help='Image filename')
    parser_infer_image.add_argument('--out', help='Output filename')
    parser_infer_image.add_argument('--network-snapshot', help='Trained network pickle')
    parser_infer_image.set_defaults(func=infer_image)

    args = parser.parse_args()
    submit_config.run_desc = desc + args.desc
    if args.run_dir_root is not None:
        submit_config.run_dir_root = args.run_dir_root
    if args.command is not None:
        args.func(args)
    else:
        # Train if no subcommand was given
        train(args)
