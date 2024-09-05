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
                "values": [1e-3, 5e-4, 1e-4]
            },
            "n_epochs": {
                "values": [10000]
            },
            "batch_size": {
                "values": [32, 64, 128, 1024]
            },
            "k": {
                "values": [4, 6, 8]
            }
        }
    }
