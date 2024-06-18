from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from pycox.models import DeepHitSingle
import torchtuples as tt
from models import CauseSpecificNet
from pycox.models import DeepHit
from auton_survival.estimators import SurvivalModel

def make_cox_model(config):
    n_iter = config['n_iter']
    tol = config['tol']
    model = CoxPHSurvivalAnalysis(alpha=0.0001, n_iter=n_iter, tol=tol)
    return model

def make_coxnet_model(config):
    l1_ratio = config['l1_ratio']
    alpha_min_ratio = config['alpha_min_ratio']
    n_alphas = config['n_alphas']
    normalize = config['normalize']
    tol = config['tol']
    max_iter = config['max_iter']
    model = CoxnetSurvivalAnalysis(fit_baseline_model=True,
                                   l1_ratio=l1_ratio,
                                   alpha_min_ratio=alpha_min_ratio,
                                   n_alphas=n_alphas,
                                   normalize=normalize,
                                   tol=tol,
                                   max_iter=max_iter)
    return model

def make_coxboost_model(config):
    n_estimators = config['n_estimators']
    learning_rate = config['learning_rate']
    max_depth = config['max_depth']
    loss = config['loss']
    min_samples_split = config['min_samples_split']
    min_samples_leaf = config['min_samples_leaf']
    max_features = config['max_features']
    dropout_rate = config['dropout_rate']
    subsample = config['subsample']
    model = GradientBoostingSurvivalAnalysis(n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            max_depth=max_depth,
                                            loss=loss,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            dropout_rate=dropout_rate,
                                            subsample=subsample,
                                            random_state=0)
    return model

def make_dcph_model(config):
    layers = config['network_layers']
    n_iter = config['n_iter']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    return SurvivalModel('dcph', random_seed=0, iters=n_iter, layers=layers,
                         learning_rate=learning_rate, batch_size=batch_size)
    
def make_dsm_model(config):
    layers = config['network_layers']
    n_iter = config['n_iter']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    return SurvivalModel('dsm', random_seed=0, iters=n_iter,
                         layers=layers, distribution='Weibull',
                         max_features='sqrt', learning_rate=learning_rate,
                         batch_size=batch_size)

def make_rsf_model(config):
    n_estimators = config['n_estimators']
    max_depth = config['max_depth']
    min_samples_split = config['min_samples_split']
    min_samples_leaf =  config['min_samples_leaf']
    max_features = config['max_features']
    model = RandomSurvivalForest(random_state=0,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features)
    return model

"""
def make_deephit_single_model(config, in_features, out_features, duration_index):
    num_nodes = config['num_nodes']
    batch_norm = config['batch_norm']
    dropout = config['dropout']
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
    model = DeepHitSingle(net, tt.optim.Adam, alpha=config['alpha'],
                          sigma=config['sigma'], duration_index=duration_index)
    model.optimizer.set_lr(config['lr'])
    return model
"""

def make_deephit_model(config, in_features, out_features, num_risks, duration_index):
    num_nodes_shared = config['num_nodes_shared']
    num_nodes_indiv = config['num_nodes_indiv']
    batch_norm = config['batch_norm']
    dropout = config['dropout']
    net = CauseSpecificNet(in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                           out_features, batch_norm, dropout)
    optimizer = tt.optim.AdamWR(lr=config['lr'],
                                decoupled_weight_decay=config['weight_decay'],
                                cycle_eta_multiplier=config['eta_multiplier'])
    model = DeepHit(net, optimizer, alpha=config['alpha'], sigma=config['sigma'],
                    duration_index=duration_index)
    return model