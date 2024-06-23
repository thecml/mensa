'''
Helper function to format params for Donna's paper
'''
def format_hyperparams(params):
    return [params['theta_layer_size'],
            params['layer_size_fine_bins'],
            [params['lr'], params['reg_constant'], params['n_batches']],
            [params['backward_c_optim'], params['hierarchical_loss'],
             params['alpha'], params['sigma']],
            params['use_theta'],
            params['use_deephit'],
            params['n_extra_bins']
            ]
    
def get_layer_size_fine_bins(dataset_name):
    if dataset_name == "als":
        return [(20, 4), (20, 4)]
    elif dataset_name == "mimic":
        return [(20, 4), (20, 4)]
    elif dataset_name == "rotterdam":
        return [(20, 3), (20, 3)]
    elif dataset_name == "seer":
        return [(50, 5), (50, 5)]
    else:
        raise ValueError("Invalid dataset name")
        
    