from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from pycox.models import DeepHitSingle
import torchtuples as tt
from models import CauseSpecificNet
from pycox.models import DeepHit

def make_cox_model(config):
    n_iter = config['n_iter']
    tol = config['tol']
    model = CoxPHSurvivalAnalysis(alpha=0.0001, n_iter=n_iter, tol=tol)
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

def make_deephit_single_model(config, in_features, out_features, duration_index):
    num_nodes = config['num_nodes']
    batch_norm = config['batch_norm']
    dropout = config['dropout']
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
    model = DeepHitSingle(net, tt.optim.Adam, alpha=config['alpha'],
                          sigma=config['sigma'], duration_index=duration_index)
    model.optimizer.set_lr(config['lr'])
    return model

def make_deephit_cr_model(config, in_features, out_features, num_risks, duration_index):
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