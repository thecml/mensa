'''
record of all hyper parameters 
format (elements in order):
    size of the layers in theta (ignored if using independent models)
    (size of layer, how many fine bins each coarse bin from the previous grain gets split into)
    learning rate, regularization constant, number of batches
    backward c index optimization, hierarchical loss, alpha, sigma for l_g
    boolean for whether to use theta (whether to joint model or not)
    boolean for whether to use deephit
    number of extra time bins (that represent t > T, for individuals who do not experience event by end of horizon) 
'''

###################################################################################################
'''
synthetic data 
'''
synthetic_hyperparams = \
{
    'direct_indep': [[20], [(20, 4), (20, 5)], [0.010, 0.01, 5], [False, False, 0.0001, 100], False, False, 0], \
    'direct_dep': [[20], [(20, 4), (20, 5)], [0.030, 0.03, 5], [False, False, 0.0001, 100], True, False, 0], \
    'direct_no_h': [[20], [(20, 4), (20, 5)], [0.010, 0.02, 5], [True, False, 0.0001, 100], True, False, 1], \
    'direct_full': [[20], [(20, 4), (20, 5)], [0.010, 0.02, 5], [True, True, 0.0001, 100], True, False, 1], \
    'hierarch_no_h': [[20], [(20, 4), (20, 5)], [0.020, 0.03, 5], [True, False, 0.0001, 100], True, False, 1], \
    'hierarch_full': [[20], [(20, 4), (20, 5)], [0.025, 0.05, 5], [True, True, 0.0001, 100], True, False, 1], \
}

###################################################################################################
'''
real data
'''
adni_hyperparams = \
{
    'direct_indep': [[500], [(100, 5), (50, 12)], [0.0001, 0.3, 5], [False, False, 0.05, 90], False, False, 0], \
    'direct_dep': [[500], [(100, 5), (50, 12)], [0.0001, 0.2, 5], [False, False, 0.05, 100], True, False, 0], \
    'direct_no_h': [[500], [(100, 5), (50, 12)], [0.0003, 0.7, 5], [True, False, 0.05, 100], True, False, 1], \
    'direct_full': [[500], [(100, 5), (50, 12)], [0.0003, 0.7, 5], [True, True, 0.05, 100], True, False, 1], \
    'hierarch_no_h': [[500], [(100, 5), (50, 12)], [0.0003, 0.5, 5], [True, False, 0.05, 100], True, False, 1], \
    'hierarch_full': [[500], [(100, 5), (50, 12)], [0.0003, 0.7, 5], [True, True, 0.05, 100], True, False, 1], \
} 

seer_hyperparams = \
{
    'direct_indep': [[100], [(50, 10), (50, 12)], [0.001, 0.001, 5], [False, False, 0.001, 10], False, False, 0], \
    'direct_dh': [[100], [(50, 10), (50, 12)], [0.001, 0.01, 5], [False, False, 0.0001, 10], True, True, 0], \
    'direct_dep': [[100], [(50, 10), (50, 12)], [0.001, 0.001, 5], [False, False, 0.001, 10], True, False, 0], \
    'direct_no_h': [[100], [(50, 10), (50, 12)], [0.0005, 0.002, 5], [True, False, 0.001, 10], True, False, 1], \
    'direct_full': [[100], [(50, 10), (50, 12)], [0.0005, 0.002, 5], [True, True, 0.001, 10], True, False, 1], \
    'hierarch_no_h': [[100], [(50, 10), (50, 12)], [0.002, 0.05, 5], [True, False, 0.0003, 10], True, False, 1], \
    'hierarch_full': [[100], [(50, 10), (50, 12)], [0.002, 0.08, 5], [True, True, 0.0009, 10], True, False, 1], \
} 

mimic_hyperparams = \
{
    'direct_indep': [[500], [(100, 6), (100, 2)], [0.0001, 0.25, 5], [False, False, 0.0005, 10], False, False, 0], \
    'direct_dep': [[500], [(100, 6), (100, 2)], [0.0001, 0.25, 5], [False, False, 0.0005, 10], True, False, 0], \
    'direct_no_h': [[500], [(100, 6), (100, 2)], [0.0007, 0.02, 5], [True, False, 0.0001, 100], True, False, 1], \
    'direct_full': [[500], [(100, 6), (100, 2)], [0.0005, 0.07, 5], [True, True, 0.0001, 500], True, False, 1], \
    'hierarch_no_h': [[500], [(100, 6), (100, 2)], [0.0005, 0.05, 5], [True, False, 0.0001, 100], True, False, 1], \
    'hierarch_full': [[500], [(100, 6), (100, 2)], [0.0003, 0.15, 5], [True, True, 0.0001, 500], True, False, 1], \
}

###################################################################################################
'''
putting everything together
'''
all_hyperparams = \
{
    'Synthetic': synthetic_hyperparams, \
    'ADNI': adni_hyperparams, \
    'SEER': seer_hyperparams, \
    'MIMIC': mimic_hyperparams \
}