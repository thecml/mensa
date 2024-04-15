import pandas as pd
import numpy as np
import config as cfg
from utility.survival import make_time_bins
from utility.evaluator import LifelinesEvaluator
from data_loader import SyntheticDataLoader
import torch
import random
import warnings
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
from utility.survival import impute_and_scale
import pycox
import torchtuples as tt
from pycox.models import DeepHitSingle
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
    config = dotdict(cfg.PARAMS_DEEPHIT_SINGLE)
    
    # Load data
    dl = SyntheticDataLoader().load_data()
    num_features, cat_features = dl.get_features()
    data_packages = dl.split_data()
    
    n_events = 2
    for event_id in range(n_events):
        train_data = [data_packages[0][0], data_packages[0][1][:,event_id], data_packages[0][2][:,event_id]]
        test_data = [data_packages[1][0], data_packages[1][1][:,event_id], data_packages[1][2][:,event_id]]
        val_data = [data_packages[2][0], data_packages[2][1][:,event_id], data_packages[2][2][:,event_id]]

        # Scale data
        #train_data[0] = impute_and_scale(train_data[0].values, norm_mode='standard')
        #val_data[0] = impute_and_scale(val_data[0].values, norm_mode='standard')        
        #test_data[0] = impute_and_scale(test_data[0].values, norm_mode='standard')

        # Convert types        
        train_data[0] = train_data[0].astype('float32')
        val_data[0] = val_data[0].astype('float32')
        test_data[0] = test_data[0].astype('float32')
        
        num_durations = config['num_durations']
        labtrans = DeepHitSingle.label_transform(num_durations)
        get_target = lambda df: (df[1], df[2])
        y_train = labtrans.fit_transform(*get_target(train_data))
        y_val = labtrans.transform(*get_target(val_data))

        train = (train_data[0], y_train)
        val = (val_data[0], y_val)

        # We don't need to transform the test labels
        durations_test, events_test = get_target(test_data)
        
        # Make model
        in_features = train_data[0].shape[1]
        num_nodes = config['num_nodes']
        out_features = labtrans.out_features
        batch_norm = config['batch_norm']
        verbose = config['verbose']
        dropout = config['dropout']
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
        
        # Train
        model = DeepHitSingle(net, tt.optim.Adam, alpha=config['alpha'],
                              sigma=config['sigma'], duration_index=labtrans.cuts)
        batch_size = config['batch_size']
        model.optimizer.set_lr(config['hr'])
        epochs = config['epochs']
        if config['early_stop']:
            callbacks = [tt.callbacks.EarlyStopping(patience=config['patience'])]
        else:
            callbacks = []
        model.fit(train_data[0], y_train, batch_size, epochs, callbacks, verbose=verbose, val_data=val)
        
        # Predict
        surv = model.predict_surv_df(test_data[0])
        
        # Evaluate
        x_test = torch.tensor(test_data[0], dtype=torch.float32, device=device)
        survival_outputs = pd.DataFrame(surv)
        lifelines_eval = LifelinesEvaluator(survival_outputs.T, test_data[1].flatten(), test_data[2].flatten(),
                                            train_data[1].flatten(), train_data[2].flatten())
        mae_hinge = lifelines_eval.mae(method="Hinge")
        ci = lifelines_eval.concordance()[0]
        d_calib = lifelines_eval.d_calibration()[0]

        print(f"Done training event {event_id}. CI={round(ci,2)}, MAE={round(mae_hinge,2)}, D-Calib={round(d_calib,2)}")

        