from typing import Tuple
import numpy as np
import pandas as pd
import numpy as np
import config as cfg
from torch.utils.data import DataLoader, TensorDataset
from utility.survival import make_stratified_split, convert_to_structured, make_time_bins, make_event_times
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from preprocessor import Preprocessor
import torch

def split_and_scale_data(df, num_features, cat_features):
    # Split data in train/valid/test sets
    df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='both', frac_train=0.7,
                                                        frac_valid=0.1, frac_test=0.2, random_state=0)
    X_train = df_train[cat_features+num_features]
    X_valid = df_valid[cat_features+num_features]
    X_test = df_test[cat_features+num_features]
    y_train = convert_to_structured(df_train["time"], df_train["event"])
    y_valid = convert_to_structured(df_valid["time"], df_valid["event"])
    y_test = convert_to_structured(df_test["time"], df_test["event"])
    
    # Scale data
    preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat='mean')
    transformer = preprocessor.fit(X_train, cat_feats=cat_features, num_feats=num_features,
                                   one_hot=True, fill_value=-1)
    X_train = transformer.transform(X_train)
    X_valid = transformer.transform(X_valid)
    X_test = transformer.transform(X_test)

    # Make event times
    time_bins = make_event_times(np.array(y_train["time"]), np.array(y_train["event"]))

    # Format data for training the NN
    data_train = X_train.copy()
    data_train["time"] = pd.Series(y_train['time'])
    data_train["event"] = pd.Series(y_train['event']).astype(int)
    data_valid = X_valid.copy()
    data_valid["time"] = pd.Series(y_valid['time'])
    data_valid["event"] = pd.Series(y_valid['event']).astype(int)
    data_test = X_test.copy()
    data_test["time"] = pd.Series(y_test['time'])
    data_test["event"] = pd.Series(y_test['event']).astype(int)
    
    return (data_train, data_valid, data_test, time_bins)

def scale_data(X_train, X_test, cat_features, num_features) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat='mean')
    transformer = preprocessor.fit(X_train, cat_feats=cat_features, num_feats=num_features,
                                   one_hot=True, fill_value=-1)
    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)
    return (X_train, X_test)