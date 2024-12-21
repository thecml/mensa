def get_mensa_sweep_cfg():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_loss",
            "goal": "minimize"
        },
        "parameters": {
            "layers": {
                "values": [[16], [32], [64], [128]]
            },
            "lr": {
                "values": [1e-2, 1e-3, 5e-4, 1e-4, 1e-5]
            },
            "n_epochs": {
                "values": [10000]
            },
            "batch_size": {
                "values": [16, 32, 64, 128]
            },
            "n_dists": {
                "values": [1, 3, 5, 10]
            },
            "dropout_rate": {
                "values": [0, 0.1, 0.25, 0.5]
            },
            'weight_decay': {
                "values": [1e-2, 1e-3, 1e-4, 1e-5]
            }
        }
    }
