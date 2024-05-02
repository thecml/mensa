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
DEEPHIT_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'deephit')
DIRECT_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'direct')
HIERARCH_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'hierarch')
SURVTRACE_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'survtrace')
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
DIRECT_FULL_PARAMS = {
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

HIERARCH_FULL_PARAMS = {
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

COX_PARAMS = {
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

COX_MULTI_PARAMS = {
    'hidden_size': 32,
    'mu_scale': None,
    'rho_scale': -5,
    'sigma1': 1,
    'sigma2': 0.002,
    'pi': 0.5,
    'verbose': False,
    'lr': 0.005,
    'c1': 0.01,
    'num_epochs': 10,
    'dropout': 0.5,
    'n_samples_train': 10,
    'n_samples_test': 100,
    'batch_size': 32,
    'early_stop': True,
    'patience': 10
}

COX_MULTI_GAUSSIAN_PARAMS = {
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
    'num_epochs': 100,
    'dropout': 0.5,
    'n_samples_train': 10,
    'n_samples_test': 100,
    'batch_size': 32,
    'early_stop': True,
    'patience': 10
}

DSM_PARAMS = {
    'network_layers': [32, 32],
    'learning_rate': 0.001,
    'n_iters' : 100
    }

RSF_PARAMS = {
    'n_estimators': 100,
    'max_depth' : None,
    'min_samples_split': 6,
    'min_samples_leaf': 3,
    'max_features': None,
    "random_state": 0
    }

MTLR_PARAMS = {
    'hidden_size': 64,
    'mu_scale': None,
    'rho_scale': -5,
    'sigma1': 1,
    'sigma2': 0.002,
    'pi': 0.5,
    'verbose': False,
    'lr': 0.00008,
    'num_epochs': 1000,
    'dropout': 0.5,
    'n_samples_train': 10,
    'n_samples_test': 100,
    'batch_size': 32,
    'early_stop': True,
    'patience': 50}

DEEPHIT_PARAMS = {
    'num_durations': 10,    
    'num_nodes_shared': [32, 32],
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

SURVTRACE_PARAMS = {
    "num_hidden_layers": 3,
    "hidden_size": 16,
    "intermediate_size": 64,
    "num_attention_heads": 2,
    "initializer_range": .02,
    "batch_size": 128,
    "weight_decay": 0,
    "learning_rate": 0.0001,
    "epochs": 100,
    "early_stop_patience": 5,
    "hidden_dropout_prob": 0,
    "seed": 0,
    "hidden_act": "gelu",
    "attention_probs_dropout_prob": 0.1,
    "layer_norm_eps": 1000000000000,
    "checkpoint": "./checkpoints/survtrace.pt",
    "max_position_embeddings": 512,
    "chunk_size_feed_forward": 0,
    "output_attentions": False,
    "output_hidden_states": False,
    "tie_word_embeddings": True,
    "pruned_heads": {}
}

PARAMS_MENSA = {
    
}