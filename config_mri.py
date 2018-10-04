# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import dnnlib
import dnnlib.submission.submit as submit

# Submit config
# ------------------------------------------------------------------------------------------

submit_config = dnnlib.SubmitConfig()
submit_config.run_dir_root = "results"
submit_config.run_dir_ignore += ['datasets', 'results']

desc = "n2n-mri"

# Tensorflow config
# ------------------------------------------------------------------------------------------

tf_config = dnnlib.EasyDict()
tf_config["graph_options.place_pruned_graph"] = True

#----------------------------------------------------------------------------
# Paths etc.

data_dir    = 'datasets'
num_gpus    = 1

#----------------------------------------------------------------------------
# Baseline configuration.

run_desc    = desc
random_seed = 1000

#----------------------------------------------------------------------------
# Basic MRI runs.

run_desc = 'mri'
train    = dict(corrupt_params=dict(), augment_params=dict())

run_desc += '-ixi'
train.update(dataset_train=dict(fn='ixi_train.pkl'), augment_params=dict(translate=64)) # 4936 images, lots of augmentation.
train.update(dataset_test=dict(fn='ixi_valid.pkl'))                     # use all images, should be 1000

train['run_func_name'] = 'train_mri.train'

train['corrupt_params'].update(type='bspec', p_at_edge=0.025)   # 256x256 avg = 0.10477
train.update(learning_rate_max=0.001)
# Noise2noise (corrupt_targets=True) or noise2clean (corrupt_targets=False)
train.update(corrupt_targets=True)
train.update(post_op='fspec')
train.update(num_epochs=300)                                    # Long training runs.

# Paper cases. Overrides post-op and target corruption modes.
if train.get('corrupt_targets'):
    run_desc += '_s-n2n_'
else:
    run_desc += '_s-n2c_'

# Final inference only. Note: verify that dataset, corruption, and post_op match with loaded network.
#train.update(load_network='382-mri-ixi_s-n2n_-lr0.001000-Cbs0.025000-At64-Pfspec/network-final.pkl', start_epoch='final')      # N2N
#train.update(load_network='380-mri-ixi_s-n2c_-lr0.001000-clean-Cbs0.025000-At64-Pfspec/network-final.pkl', start_epoch='final') # N2C

if train.get('num_epochs'): run_desc += '-ep%d' % train['num_epochs']
if train.get('learning_rate_max'): run_desc += '-lr%f' % train['learning_rate_max']
if not train.get('corrupt_targets', True): run_desc += '-clean'
if train.get('minibatch_size'): run_desc += '-mb%d' % train['minibatch_size']
if train['corrupt_params'].get('type') == 'gaussian': run_desc += '-Cg%f' % train['corrupt_params']['scale']    
if train['corrupt_params'].get('type') == 'bspec': run_desc += '-Cbs%f' % train['corrupt_params']['p_at_edge']
if train['corrupt_params'].get('type') == 'bspeclin': run_desc += '-Cbslin%f' % train['corrupt_params']['p_at_edge']
if train['augment_params'].get('translate', 0) > 0: run_desc += '-At%d' % train['augment_params']['translate']
if train.get('post_op'): run_desc += '-P%s' % train['post_op']
if random_seed != 1000: run_desc += '-%d' % random_seed
if train.get('load_network'): run_desc += '-LOAD%s' % train['load_network'][:3]
if train.get('start_epoch'): run_desc += '-start%s' % train['start_epoch']

# Farm submit config
# ----------------------------------------------------------------

# Number of GPUs
run_desc += "-1gpu"
submit_config.num_gpus = 1

# Submission target
run_desc += "-L"; submit_config.submit_target = dnnlib.SubmitTarget.LOCAL

submit_config.run_desc = run_desc


#----------------------------------------------------------------------------
if __name__ == "__main__":
    submit.submit_run(submit_config, **train)
