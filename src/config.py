from pathlib import Path

# Directories
ROOT_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = Path.joinpath(ROOT_DIR, "data")
CONFIGS_DIR = Path.joinpath(ROOT_DIR, 'configs')
DATA_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'data')
HIERARCH_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'hierarch')
DGP_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'dgp')
MENSA_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'mensa')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')
PLOTS_DIR = Path.joinpath(ROOT_DIR, 'plots')
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')

# This contains default parameters for the models
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
    'verbose': False
}

COXPH_PARAMS = {
    'alpha': 0,
    'ties': 'breslow',
    'n_iter': 100,
    'tol': 1e-9
}

COXBOOST_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 1,
    'loss': 'coxph',
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'dropout_rate': 0.0,
    'subsample': 0.8,
    'seed': 0,
    'test_size': 0.3,
}

DEEPSURV_PARAMS = {
    'hidden_size': 100,
    'verbose': False,
    'lr': 0.001,
    'c1': 0.01,
    'num_epochs': 1000,
    'dropout': 0.25,
    'early_stop': True,
    'patience': 10
}

MTLR_PARAMS = {
    'verbose': False,
    'lr': 0.001,
    'c1': 0.01,
    'num_epochs': 1000,
    'dropout': 0.25,
    'batch_size': 32,
    'early_stop': True,
    'patience': 10
}

RSF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 1,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    "random_state": 0
}

DSM_PARAMS = {
    'network_layers': [32],
    'learning_rate': 0.001,
    'n_iter': 10000,
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
    'patience': 10,
}

MTLRCR_PARAMS = {
    'verbose': False,
    'lr': 1e-3,
    'c1': 0.01,
    'num_epochs': 1000,
    'dropout': 0.25,
    'batch_size': 32,
    'early_stop': True,
    'patience': 10
}

DCSURVIVAL_PARAMS = {
    'depth': 2,
    'widths': [100, 100],
    'lc_w_range': [0, 1.0],
    'shift_w_range': [0., 2.0],
    'learning_rate': 1e-4
}

MENSA_PARAMS = {
    'layers': [32],
    'lr': 0.001,
    'n_epochs': 1000,
    'n_dists': 3,
    'batch_size': 32,
    'weight_decay': 0,
    'dropout_rate': 0.25,
}