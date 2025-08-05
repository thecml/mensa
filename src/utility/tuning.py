import numpy as np

def get_mensa_sweep_cfg():
    return {
        "method": "bayes",
        "metric": {
            "name": "c_harrell",
            "goal": "maximize"
        },
        "parameters": {
            "layers": {
                "values": [[16], [32], [64], [128]]
            },
            "lr": {
                "values": [1e-2, 1e-3, 5e-4, 1e-4, 1e-5]
            },
            "n_epochs": {
                "values": [1000]
            },
            "batch_size": {
                "values": [32]
            },
            "n_dists": {
                "values": [1, 3, 5]
            },
            "dropout_rate": {
                "values": [0, 0.1, 0.25, 0.5]
            },
            'weight_decay': {
                "values": [0, 1e-3, 1e-4, 1e-5]
            }
        }
    }

def get_coxboost_sweep_cfg():
    return {
        "method": "bayes",
        "metric": {
            "name": "c_harrell",
            "goal": "maximize"
        },
        "parameters": {
            "n_estimators": {
                "values": [50, 100, 200, 400, 600, 800, 1000]
            },
            "learning_rate": {
                "values": [0.01, 0.05, 0.1, 1.0]
            },
            "max_depth": {
                "values": [3, 5, 10]
            },
            "loss": {
                "values": ['coxph']
            },
            "min_samples_split": {
                "values": [2, 5, 10]
            },
            "max_features": {
                "values": [None, "auto", "sqrt", "log2"]
            },
            "dropout_rate": {
                "values": [0, 0.25, 0.5]
            },
            "subsample": {
                "values": [0.1, 0.25, 0.5, 1]
            }
        }
    }

def get_coxph_sweep_cfg():
    return {
        "method": "bayes",
        "metric": {
            "name": "c_harrell",
            "goal": "maximize"
        },
        "parameters": {
            "alpha": {
                "values": [0.0, 0.01, 0.1, 1.0, 10.0]
            },
            "ties": {
                "values": ["breslow", "efron"]
            },
            "n_iter": {
                "values": [50, 100, 200, 500]
            },
            "tol": {
                "values": [1e-5, 1e-7, 1e-9]
            }
        }
    }
    
def get_rsf_sweep_cfg():
    return {
        "method": "bayes",
        "metric": {
            "name": "c_harrell",
            "goal": "maximize"
        },
        "parameters": {
            "n_estimators": {
                "values": [100, 200, 500, 1000]
            },
            "max_depth": {
                "values": [3, 5, 10]
            },
            "min_samples_split": {
                "values": [2, 5, 10]
            },
            "min_samples_leaf": {
                "values": [1, 5, 10]
            },
            "max_features": {
                "values": ["sqrt", "log2", None]
            },
            "bootstrap": {
                "values": [True, False]
            }
        }
    }
    
def get_deepsurv_sweep_cfg():
    return {
        "method": "bayes",
        "metric": {
            "name": "c_harrell",
            "goal": "maximize"
        },
        "parameters": {
            "hidden_size": {
                "values": [32, 64, 100, 128]
            },
            "lr": {
                "values": [1e-4, 5e-4, 1e-3, 5e-3]
            },
            "c1": {
                "values": [0.0, 0.001, 0.01, 0.1]
            },
            "dropout": {
                "values": [0.0, 0.1, 0.25, 0.5]
            },
            "num_epochs": {
                "values": [500, 1000, 2000]
            },
            "patience": {
                "values": [5, 10, 20]
            }
        }
    }

def get_deephit_sweep_cfg():
    return {
        "method": "bayes",
        "metric": {
            "name": "c_harrell",
            "goal": "maximize"
        },
        "parameters": {
            "num_nodes_shared": {
                "values": [16, 32, 64, 128]
            },
            "num_nodes_indiv": {
                "values": [16, 32, 64, 128]
            },
            "dropout": {
                "values": [0.0, 0.1, 0.25, 0.5]
            },
            "alpha": {
                "values": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            },
            "sigma": {
                "values": [0.01, 0.05, 0.1, 0.2, 0.5]
            },
            "batch_size": {
                "values": [32]
            },
            "lr": {
                "values": [1e-4, 5e-4, 1e-3, 5e-3]
            },
            "weight_decay": {
                "values": [0.0, 0.001, 0.01, 0.1]
            },
            "eta_multiplier": {
                "values": [0.5, 0.8, 1.0, 1.2]
            },
            "epochs": {
                "values": [1000]
            },
            "patience": {
                "values": [10]
            }
        }
    }

def get_hierarch_sweep_cfg():
    return {
        "method": "bayes",
        "metric": {
            "name": "c_harrell",
            "goal": "maximize"
        },
        "parameters": {
            "theta_layer_size": {
                "values": [[50], [100], [200]]
            },
            "layer_size_fine_bins": {
                "values": [
                    [(50, 5)],
                    [(100, 10)],
                    [(150, 10)]
                ]
            },
            "lr": {
                "values": [1e-4, 5e-4, 1e-3]
            },
            "reg_constant": {
                "values": [0.0, 0.01, 0.05, 0.1]
            },
            "n_batches": {
                "values": [10]
            },
            "batch_size": {
                "values": [32]
            },
            "alpha": {
                "values": [0.0, 1e-5, 1e-4, 1e-3]
            },
            "sigma": {
                "values": [1, 5, 10, 20]
            },
            "n_extra_bins": {
                "values": [1]
            },
            "use_theta": {
                "values": [True]
            },
            "use_deephit": {
                "values": [False]
            },
            "hierarchical_loss": {
                "values": [True]
            },
            "backward_c_optim": {
                "values": [False]
            }
        }
    }

def get_mtlr_sweep_cfg():
    return {
        "method": "bayes",
        "metric": {
            "name": "c_harrell",
            "goal": "maximize"
        },
        "parameters": {
            "lr": {
                "values": [1e-4, 5e-4, 1e-3, 5e-3]
            },
            "c1": {
                "values": [0.0, 0.001, 0.01, 0.1]
            },
            "dropout": {
                "values": [0.0, 0.1, 0.25, 0.5]
            },
            "batch_size": {
                "values": [32]
            },
            "num_epochs": {
                "values": [1000]
            },
            "patience": {
                "values": [10]
            }
        }
    }

def get_dsm_sweep_cfg():
    return {
        "method": "bayes",
        "metric": {
            "name": "c_harrell",
            "goal": "maximize"
        },
        "parameters": {
            "network_layers": {
                "values": [
                    [32],
                    [64],
                    [128]
                ]
            },
            "learning_rate": {
                "values": [1e-4, 5e-4, 1e-3, 5e-3]
            },
            "n_iter": {
                "values": [5000, 10000, 20000]
            },
            "batch_size": {
                "values": [16, 32, 64]
            },
            "k": {
                "values": [1, 3, 5]
            }
        }
    }
