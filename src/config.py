from pathlib import Path
import numpy as np

# Directories
ROOT_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = Path.joinpath(ROOT_DIR, "data")
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
CONFIGS_DIR = Path.joinpath(ROOT_DIR, 'configs')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')
COX_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'cox')
COXBOOST_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'coxboost')
RSF_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'rsf')
MTLR_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'mtlr')
DEEPCR_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'deepcr')
DEEPHIT_SINGLE_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'deephit-single')
DEEPHIT_CR_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'deephit-comp')
DIRECT_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'direct')
HIERARCH_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'hierarch')
DATASET_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'dataset')
MENSA_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'mensa')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')

# This contains DEFAULT parameters for the models
'''
record of all hyper parameters 
format (elements in order):
    size of the layers in theta (ignored if using independent models)
    (size of layer, how many fine bins each coarse bin from the previous grain gets split into)
    learning rate, regularization constant, number of batches
    backward c index optimization, hierarchical loss, alpha, sigma for l_g
    boolean for whether to use theta (whether to joint model or not)
    boolean for whether to use deephit
    number of extra time bins (that represent t > T, for individuals who do not experience event by end of horizon) 
'''
PARAMS_DIRECT_FULL = {
    'theta_layer_size': [20], # size of the shared layer
    'layer_size_fine_bins': [(20, 4), (20, 5)], #product of the second number has to equal the number of bins
    'lr': 0.010, # ranges 0.0001 - 0.01
    'reg_constant': 0.02, # 0.01, 0.02, 0.05
    'n_batches': 5, # batch size = trainset/n_batches
    'backward_c_optim': False, # global
    'hierarchical_loss': False,
    'alpha': 0.0001, # 0.0001, 0.05, 0.0005, and 0.001
    'sigma': 100, #10, 100, 1000
    'use_theta': True,
    'use_deephit': False,
    'n_extra_bins': 1,
    'verbose': True
}

PARAMS_HIERARCH_FULL = {
    'theta_layer_size': [20],
    'layer_size_fine_bins': [(20, 4), (20, 5)],
    'lr': 0.025,
    'reg_constant': 0.05,
    'n_batches': 5,
    'backward_c_optim': False,
    'hierarchical_loss': True,
    'alpha': 0.0001,
    'sigma': 100,
    'use_theta': True,
    'use_deephit': False,
    'n_extra_bins': 1,
    'verbose': True
}

PARAMS_COX = {
    'hidden_size': 32,
    'mu_scale': None,
    'rho_scale': -5,
    'sigma1': 1,
    'sigma2': 0.002,
    'pi': 0.5,
    'verbose': False,
    'lr': 0.005,
    'c1': 0.01,
    'num_epochs': 50,
    'dropout': 0.5,
    'n_samples_train': 10,
    'n_samples_test': 100,
    'batch_size': 32,
    'early_stop': True,
    'patience': 10
}

PARAMS_COX_MULTI = {
    'hidden_size': 32,
    'mu_scale': None,
    'rho_scale': -5,
    'sigma1': 1,
    'sigma2': 0.002,
    'pi': 0.5,
    'verbose': False,
    'lr': 0.005,
    'c1': 0.01,
    'num_epochs': 50,
    'dropout': 0.5,
    'n_samples_train': 10,
    'n_samples_test': 100,
    'batch_size': 32,
    'early_stop': True,
    'patience': 10
}

PARAMS_COX_MULTI_GAUSSIAN = {
    'hidden_size': 32,
    'mu_scale': None,
    'rho_scale': -5,
    'sigma1': 1,
    'sigma2': 0.002,
    'pi': 0.5,
    'verbose': False,
    'lr': 0.005,
    'c1': 0.01,
    'num_epochs': 50,
    'dropout': 0.5,
    'n_samples_train': 10,
    'n_samples_test': 100,
    'batch_size': 32,
    'early_stop': True,
    'patience': 10
}

PARAMS_MTLR = {
    'hidden_size': 32,
    'mu_scale': None,
    'rho_scale': -5,
    'sigma1': 1,
    'sigma2': 0.002,
    'pi': 0.5,
    'verbose': False,
    'lr': 0.00008,
    'c1': 0.01,
    'num_epochs': 100,
    'dropout': 0.5,
    'n_samples_train': 10,
    'n_samples_test': 100,
    'batch_size': 32,
    'early_stop': True,
    'patience': 10
}

PARAMS_DEEPHIT_SINGLE = {
    'num_nodes': [32, 32],
    'num_durations': 10,
    'batch_norm': True,
    'verbose': False,
    'dropout': 0.1,
    'alpha': 0.2,
    'sigma': 0.1,
    'batch_size': 32,
    'lr': 0.01,
    'epochs': 100,
    'early_stop': True,
    'patience': 10,
}

PARAMS_DEEPHIT_MULTI = {
    'num_durations': 10,    
    'num_nodes_shared': [64, 64],
    'num_nodes_indiv': [32],
    'batch_norm': True,
    'verbose': False,
    'dropout': 0.1,
    'alpha': 0.2,
    'sigma': 0.1,
    'batch_size': 32,
    'lr': 0.01,
    'weight_decay': 0.01,
    'eta_multiplier': 0.8,
    'epochs': 100,
    'early_stop': True,
    'patience': 10,
}

PARAMS_MENSA = {
    
}