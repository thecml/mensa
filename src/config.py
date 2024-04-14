from pathlib import Path
import numpy as np

# Directories
ROOT_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = Path.joinpath(ROOT_DIR, "data")
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
CONFIGS_DIR = Path.joinpath(ROOT_DIR, 'configs')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')
COX_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'cox')
MTLR_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'mtlr')
DEEPCR_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'deepcr')
DEEPHIT_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'deephit')
HIERACH_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'hierach')
MENSA_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'mensa')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')

# This contains DEFAULT parameters for the models
'''
Record of all settings for datasets
Definitions:
    location: where the data is stored
    features: where in location (see above) the covariates/features are stored
    terminal event: event such that no other events can occur after it
    discrete: whether the time values are discrete
    event ranks: each key represents and event, the values are the events that prevent it
    event groups: each key represents the position in a trajectory (e.g., first, second, ...), values represent which events can occur in that position
    min_time: earliest event time
    max_time: latest event time (prediction horizon)
    min_epoch: minimum number of epochs to train for (while learning the model)
'''
SYNTHETIC_DATA_SETTINGS = {
    'num_events': 2,
    'num_bins': 20,
    'terminal_events': [1],
    'discrete': False,
    'event_ranks': {0:[], 1:[]},
    'event_groups': {0:[0, 1], 1:[0, 1]},
    'min_time': 0,
    'max_time': 20,
    'min_epoch': 50,
}

PARAMS_DIRECT_FULL = {
    'theta_layer_size': [20],
    'layer_size_fine_bins': [(20, 4), (20, 5)],
    'lr': 0.010,
    'reg_constant': 0.02,
    'n_batches': 5,
    'backward_c_optim': True,
    'hierarchical_loss': True,
    'alpha': 0.0001,
    'sigma': 100,
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
    'backward_c_optim': True,
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