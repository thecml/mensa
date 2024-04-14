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