import numpy as np

def get_mensa_sweep_cfg():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_loss",
            "goal": "minimize"
        },
        "parameters": {
            "layers": {
                "values": [[16], [32], [64], [64, 128], [32, 32],
                           [32, 128], [32, 32, 32], [64, 64],
                           [128], [128, 128]]
            },
            "lr": {
                "values": [0.01, 0.005, 0.001, 0.0001]
            },
            "dropout":{
                "values": [0, 0.1, 0.2, 0.25, 0.5, 0.6, 0.75]
            },
            "n_epochs": {
                "values": [100, 200, 250, 500, 750, 1000, 2000, 3000]
            },
            "batch_size": {
                "values": [128]
            },
            "activation_fn": {
                "values": ["relu"]
            },
            "optimizer": {
                "values": ["adam", "adamw"]
            }
        }
    }
