from pathlib import Path
import numpy as np

# Directories
ROOT_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = Path.joinpath(ROOT_DIR, "data")
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
CONFIGS_DIR = Path.joinpath(ROOT_DIR, 'configs')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')
DATA_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'data')
HIERARCH_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'hierarch')
DGP_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'dgp')
MENSA_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'mensa')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')

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
SYNTHETIC_SETTINGS = {
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
HIERARCH_PARAMS = {
    'theta_layer_size': [100],
    'layer_size_fine_bins': [(50, 5), (50, 5)],
    'lr': 0.001,
    'reg_constant': 0.05,
    'n_batches': 10,
    'batch_size': 32,
    'backward_c_optim': False,
    'hierarchical_loss': True,
    'alpha': 0.0001,
    'sigma': 10,
    'use_theta': True,
    'use_deephit': False,
    'n_extra_bins': 1,
    'verbose': True
}

COX_PARAMS = {
    'alpha': 0,
    'ties': 'breslow',
    'n_iter': 100,
    'tol': 1e-9
}

COXNET_PARAMS = {
    'l1_ratio': 1,
    'alpha_min_ratio': 0.1,
    'n_alphas': 100,
    'normalize': False,
    'tol': 0.1,
    'max_iter': 100000
}

COXBOOST_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'loss': 'coxph',
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': None,
    'dropout_rate': 0.0,
    'subsample': 1.0,
    'seed': 0,
    'test_size': 0.3,
}

DEEPSURV_PARAMS = {
    'hidden_size': 32,
    'mu_scale': None,
    'rho_scale': -5,
    'sigma1': 1,
    'sigma2': 0.002,
    'pi': 0.5,
    'verbose': False,
    'lr': 0.005,
    'c1': 0.01,
    'num_epochs': 1000,
    'dropout': 0.25,
    'n_samples_train': 10,
    'n_samples_test': 100,
    'batch_size': 32,
    'early_stop': True,
    'patience': 10
}

MTLR_PARAMS = {
    'hidden_size': 32,
    'mu_scale': None,
    'rho_scale': -5,
    'sigma1': 1,
    'sigma2': 0.002,
    'pi': 0.5,
    'verbose': False,
    'lr': 0.00008,
    'c1': 0.01,
    'num_epochs': 1000,
    'dropout': 0.5,
    'n_samples_train': 10,
    'n_samples_test': 100,
    'batch_size': 32,
    'early_stop': True,
    'patience': 10
}

RSF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 3,
    'min_samples_split': 6,
    'min_samples_leaf': 3,
    'max_features': None,
    "random_state": 0
}

DSM_PARAMS = {
    'network_layers': [32],
    'learning_rate': 0.001,
    'n_iter': 1000,
    'batch_size': 32
}

DEEPHIT_PARAMS = {
    'num_nodes_shared': [32],
    'num_nodes_indiv': [32],
    'batch_norm': True,
    'verbose': False,
    'dropout': 0.25,
    'alpha': 0.2,
    'sigma': 0.1,
    'batch_size': 32,
    'lr': 0.001,
    'weight_decay': 0.01,
    'eta_multiplier': 0.8,
    'epochs': 1000,
    'early_stop': True,
    'patience': 100,
}

MTLRCR_PARAMS = {
    'hidden_size': 32,
    'mu_scale': None,
    'rho_scale': -5,
    'sigma1': 1,
    'sigma2': 0.002,
    'pi': 0.5,
    'verbose': False,
    'lr': 1e-3,
    'c1': 0.01,
    'num_epochs': 1000,
    'dropout': 0.5,
    'n_samples_train': 10,
    'n_samples_test': 100,
    'batch_size': 32,
    'early_stop': True,
    'patience': 10
}

SURVTRACE_PARAMS = {
    "num_hidden_layers": 1,
    "hidden_size": 32,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "initializer_range": .02,
    "batch_size": 128,
    "weight_decay": 0,
    "learning_rate": 1e-4,
    "epochs": 100,
    "early_stop_patience": 10,
    "hidden_dropout_prob": 0.25,
    "seed": 0,
    "hidden_act": "gelu",
    "attention_probs_dropout_prob": 0.25,
    "layer_norm_eps": 1000000000000,
    "checkpoint": "./checkpoints/survtrace.pt",
    "max_position_embeddings": 512,
    "chunk_size_feed_forward": 0,
    "output_attentions": False,
    "output_hidden_states": False,
    "tie_word_embeddings": True,
    "pruned_heads": {}
}

DCSURVIVAL_PARAMS = {
    'depth': 2,
    'widths': [100, 100],
    'lc_w_range': [0, 1.0],
    'shift_w_range': [0., 2.0]
}
