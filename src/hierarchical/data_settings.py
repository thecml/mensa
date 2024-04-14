'''
record of all settings for datasets
definitions:
    location: where the data is stored
    features: where in location (see above) the covariates/features are stored
    terminal event: event such that no other events can occur after it
    discrete: whether the time values are discrete
    event ranks: each key represents and event, the values are the events that prevent it
    event groups: each key represents the position in a trajectory (e.g., first, second, ...), values represent which events can occur in that position
    min_time: earliest event time
    max_time: latest event time (prediction horizon)
    min_epoch: minimum number of epochs to train for (while learning the model)
'''

synthetic_settings = \
{
    'num_events': 2, \
    'num_bins': 20, \
    'terminal_events': [1], \
    'discrete': False, \
    'event_ranks': {0:[], 1:[]}, \
    'event_groups': {0:[0, 1], 1:[0, 1]}, \
    'min_time': 0, \
    'max_time': 20, \
    'min_epoch': 50, \
} 

adni_settings = \
{
    'location': '', \
    'num_events': 2, \
    'num_bins': 60, \
    'terminal_events': [1], \
    'discrete':True, \
    'event_ranks': {0:[1], 1:[]}, \
    'event_groups': {0:[0, 1]}, \
    'min_time': 0, \
    'max_time': 60, \
    'min_epoch': 50, \
}

seer_settings = \
{
    'location': '', \
    'num_events': 2, \
    'num_bins': 120, \
    'terminal_events': [0, 1], \
    'discrete':True, \
    'event_ranks': {0:[1], 1:[0]}, \
    'event_groups': {0:[0, 1]}, \
    'min_time': 0, \
    'max_time': 120, \
    'min_epoch': 50, \
}

mimic_settings = \
{
    'location': '', \
    'features': '', \
    'num_events': 3, \
    'num_bins': 12, \
    'terminal_events': [2], \
    'discrete':True, \
    'event_ranks': {0:[2], 1:[2], 2:[]}, \
    'event_groups': {0:[0, 1, 2], 1:[2]}, \
    'min_time': 0, \
    'max_time': 12, \
    'min_epoch': 30, \
}

##########################################################################################################
all_settings = \
{
    'Synthetic': synthetic_settings, \
    'ADNI': adni_settings, \
    'SEER': seer_settings, \
    'MIMIC': mimic_settings \
}