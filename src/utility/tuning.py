import numpy as np

def get_hierarch_sweep_cfg():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "theta_layer_size": {
                "values": [[32], [64], [128]]
            },
            "lr": {
                "values": [0.01, 0.001, 0.0001]
            },
            "reg_constant": {
                "values": [0.01, 0.02,0.05]
            },
            "n_batches": {
                "values": [5, 10, 20]
            },
            "alpha": {
                "values": [0.0001, 0.05, 0.0005, 0.001]
            },
            "sigma": {
                "values": [10, 100, 1000]
            }
        }
    }

def get_direct_sweep_cfg():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "theta_layer_size": {
                "values": [[32], [64], [128]]
            },
            "lr": {
                "values": [0.01, 0.001, 0.0001]
            },
            "reg_constant": {
                "values": [0.01, 0.02,0.05]
            },
            "n_batches": {
                "values": [5, 10, 20]
            },
            "alpha": {
                "values": [0.0001, 0.05, 0.0005, 0.001]
            },
            "sigma": {
                "values": [10, 100, 1000]
            }
        }
    }

def get_survtrace_sweep_cfg():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "num_hidden_layers": {
                "values": [2, 4, 6]
            },
            "hidden_size": {
                "values": [16, 32, 64]
            },
            "intermediate_size": {
                "values": [16, 32, 64]
            },
            "batch_size": {
                "values": [32, 64, 128]
            },
            "learning_rate": {
                "values": [0.01, 0.001, 0.0001]
            },
            "epochs": {
                "values": [10, 50, 100]
            },
            "early_stop_patience": {
                "values": [0]
            },
            "hidden_dropout_prob": {
                "values": [0.1, 0.25, 0.5]
            }
        }
    }

def get_deephit_sweep_cfg():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "num_nodes_shared": {
                "values": [[32], [32, 32], [64], [64, 64]]
            },
            "num_nodes_indiv": {
                "values": [16, 32, 64, 128]
            },
            "dropout": {
                "values": [0.1, 0.25, 0.5]
            },
            "batch_size": {
                "values": [32, 64, 128]
            },
            "lr": {
                "values": [0.01, 0.005, 0.001]
            },
            "early_stop": {
                "values": [False]
            },            
            "patience": {
                "values": [0]
            },
            "epochs": {
                "values": [10, 50, 100]
            },
        }
    }


def get_mtlr_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "hidden_size": {
                "values": [32, 64, 128]
            },
            "dropout": {
                "values": [0, 0.25, 0.5]
            },
            "num_epochs": {
                "values": [100, 500, 1000]
            },
            "early_stop": {
                "values": [False]
            },
            "patience": {
                "values": [0]
            },
            "batch_size": {
                "values": [32, 64, 128]
            }
        }
    }

def get_baycox_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "hidden_size": {
                "values": [32, 64, 128]
            },
            "dropout": {
                "values": [0, 0.25, 0.5]
            },
            "num_epochs": {
                "values": [100, 500, 1000]
            },
            "early_stop": {
                "values": [False]
            },
            "batch_size": {
                "values": [32, 64, 128]
            },
            "patience": {
                "values": [0]
            },
        }
    }

def get_coxboost_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "n_estimators": {
                "values": [100, 200, 400]
            },
            "learning_rate": {
                "values": [0.1]
            },
            "max_depth": {
                "values": [3, 5, 7]
            },
            "loss": {
                "values": ['coxph']
            },
            "min_samples_split": {
                "values": [float(x) for x in np.linspace(0.1, 0.9, 5, endpoint=True)]
            },
            "min_samples_leaf": {
                "values": [float(x) for x in np.linspace(0.1, 0.5, 5, endpoint=True)]
            },
            "max_features": {
                "values": [None, "auto", "sqrt", "log2"]
            },
            "dropout_rate": {
                "values": [float(x) for x in np.linspace(0.0, 0.5, 10, endpoint=True)]
            },
            "subsample": {
                "values": [float(x) for x in np.linspace(0.1, 1.0, 10, endpoint=True)]
            }
        }
    }

def get_mlp_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3
        },
        "parameters": {
            "network_layers": {
                "values": [[16], [16, 16], [32], [32, 32], [32, 32, 32],
                           [64], [32, 64], [32, 64, 64],
                           [128], [64, 128], [32, 64, 128]]
            },
            "learning_rate": {
                "values": [0.001, 0.005, 0.01]
            },
            "weight_decay": {
                "values": [1e-3, 1e-4, 1e-5, None]
            },
            "optimizer": {
                "values": ["Adam"]
            },
            "activation_fn": {
                "values": ["relu"]
            },
            "dropout": {
                "values": [0.1, 0.2, 0.25]
            },
            "batch_size": {
                "values": [32, 64, 128]
            },
            "num_epochs": {
                "values": [100]
            },
            "l2_reg": {
                "values": [0.001]
            },
            "early_stop": {
                "values": [True]
            },
            "patience": {
                "values": [10]
            },
        }
    }

def get_rsf_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
            },
        "parameters": {
            "n_estimators": {
                "values": [100, 200, 400]
            },
            "max_depth": {
                "values": [3, 5, 7]
            },
            "min_samples_split": {
                "values": [float(x) for x in np.linspace(0.1, 0.9, 5, endpoint=True)]
            },
            "min_samples_leaf": {
                "values": [float(x) for x in np.linspace(0.1, 0.5, 5, endpoint=True)]
            },
            "max_features": {
                "values": [None, 'auto', 'sqrt', 'log2']
            },
        }
    }

def get_cox_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "n_iter": {
                "values": [10, 50, 100]
            },
            "tol": {
                "values": [1e-1, 1e-5, 1e-9]
            }
        }
    }

def get_dsm_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "network_layers": {
                "values": [[32], [32, 32], [32, 32, 32],
                           [64], [32, 64], [32, 64, 64],
                           [128], [64, 128], [32, 64, 128]]
            },
            "n_iter": {
                "values": [50, 100, 200, 500, 1000]
            },
            "learning_rate": {
                "values": [0.001, 0.005, 0.01, 0.05, 0.1]
            }
        }
    }