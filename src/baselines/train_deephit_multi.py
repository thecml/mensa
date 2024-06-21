import pandas as pd
import numpy as np
import config as cfg
from utility.survival import make_time_bins
import torch
import random
import warnings
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
from utility.survival import preprocess_data
import pycox
import torchtuples as tt
from sota_models import CauseSpecificNet
from pycox.models import DeepHitSingle
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models import DeepHit
from utility.data import dotdict

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

if __name__ == "__main__":
    # Load config
    config = dotdict(cfg.DEEPHIT_PARAMS)
    
    # Load data
    url = 'https://raw.githubusercontent.com/chl8856/DeepHit/master/sample%20data/SYNTHETIC/synthetic_comprisk.csv'
    df_train = pd.read_csv(url)
    df_test = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_test.index)
    df_val = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_val.index)
    
    get_x = lambda df: (df
                        .drop(columns=['time', 'label', 'true_time', 'true_label'])
                        .values.astype('float32'))
    
    x_train = get_x(df_train)
    x_val = get_x(df_val)
    x_test = get_x(df_test)

    class LabTransform(LabTransDiscreteTime):
        def transform(self, durations, events):
            durations, is_event = super().transform(durations, events > 0)
            events[is_event == 0] = 0
            return durations, events.astype('int64')
    
    num_durations = config['num_durations']
    labtrans = LabTransform(num_durations)
    get_target = lambda df: (df['time'].values, df['label'].values)
    
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))
    durations_train, events_train = get_target(df_train)
    durations_test, events_test = get_target(df_test)
    val = (x_val, y_val)
    
    # Make net
    in_features = x_train.shape[1]
    num_nodes_shared = config['num_nodes_shared']
    num_nodes_indiv = config['num_nodes_indiv']
    num_risks = y_train[1].max()
    out_features = len(labtrans.cuts)
    batch_norm = config['batch_norm']
    dropout = config['dropout']
    net = CauseSpecificNet(in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                           out_features, batch_norm, dropout)
        
    # Train net
    optimizer = tt.optim.AdamWR(lr=config['lr'],
                                decoupled_weight_decay=config['weight_decay'],
                                cycle_eta_multiplier=config['eta_multiplier'])
    model = DeepHit(net, optimizer, alpha=config['alpha'], sigma=config['sigma'],
                    duration_index=labtrans.cuts)
    epochs = config['epochs']
    batch_size = config['batch_size']
    verbose = config['verbose']
    if config['early_stop']:
        callbacks = [tt.callbacks.EarlyStopping(patience=config['patience'])]
    else:
        callbacks = []
    model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val)
    
    # Predict
    # The survival function obtained with predict_surv_df is the probability of surviving any of the events,
    # and does, therefore, not distinguish between the event types. This means that we evaluate this "single-event case" as before.
    surv = model.predict_surv_df(x_test)
    survival_outputs = pd.DataFrame(surv.T)
    lifelines_eval = LifelinesEvaluator(survival_outputs.T, durations_test, events_test != 0,
                                        durations_train, events_train != 0)
    mae_hinge = lifelines_eval.mae(method="Hinge")
    ci = lifelines_eval.concordance()[0]
    d_calib = lifelines_eval.d_calibration()[0]
    
    print(f"Done training. CI={round(ci,2)}, MAE={round(mae_hinge,2)}, D-Calib={round(d_calib,2)}")


    