'''
Helper functions to format params for Donna's paper
'''
def format_hierarchical_hyperparams(params):
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
        
def calculate_flops(num_features, num_events, main_layers, event_layers, time_bins):
    # Ensure the first layer of the main network uses num_features as its input size
    if main_layers[0][0] != num_features:
        raise ValueError("The input size of the first layer in main_layers must match num_features.")

    # Calculate FLOPs for main network
    main_flops = sum(2 * inp * out for inp, out in main_layers)

    # Calculate FLOPs for one event network
    event_network_flops = sum(2 * inp * out for inp, out in event_layers)

    # Total event networks FLOPs
    total_event_flops = num_events * event_network_flops

    # Softmax FLOPs
    main_softmax_flops = 2 * main_layers[-1][1]  # Softmax over the last layer's output
    event_softmax_flops = num_events * 2 * time_bins  # Softmax for each event network
    total_softmax_flops = main_softmax_flops + event_softmax_flops

    # Total FLOPs
    total_flops = main_flops + total_event_flops + total_softmax_flops

    # Return detailed breakdown and total FLOPs
    return total_flops