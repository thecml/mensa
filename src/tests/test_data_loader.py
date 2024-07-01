from data_loader import (SingleEventSyntheticDataLoader,
                         CompetingRiskSyntheticDataLoader,
                         MultiEventSyntheticDataLoader,
                         MimicDataLoader)
from utility.config import load_config
import config as cfg
import random
import torch
import numpy as np

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

if __name__ == '__main__':
    data_config = load_config(cfg.DGP_CONFIGS_DIR, f"synthetic.yaml")
    dl = SingleEventSyntheticDataLoader().load_data(data_config, k_tau=0.25)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2)
    print(train_dict['X'].shape)
    
    dl = CompetingRiskSyntheticDataLoader().load_data(data_config, k_tau=0.25)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2)
    print(train_dict['X'].shape)
    
    dl = MultiEventSyntheticDataLoader().load_data(data_config, k_taus=[0.25, 0.25, 0.25])
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2)
    print(train_dict['X'].shape)
    