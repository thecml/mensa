import os
import random
from data_loader import SingleEventSyntheticDataLoader
from hierarchical import util_torch
from hierarchical.helper import format_hierarchical_hyperparams
from mensa.model import MENSA
import numpy as np
import pandas as pd
import config as cfg
from sota_models import DeepSurv, make_deephit_single, make_dsm_model
import torch
from utility.config import load_config
from utility.data import calculate_layer_size_hierarch, dotdict, format_hierarchical_data_cr
from utility.mtlr import mtlr
from utility.survival import make_time_bins
from data_loader import SingleEventSyntheticDataLoader
from fvcore.nn import FlopCountAnalysis

import logging
logging.basicConfig(level=logging.ERROR)

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

dtype = torch.float32
torch.set_default_dtype(dtype)

device = torch.device('cpu')

model_names = ["deepsurv", "deephit", "mtlr", "dsm", "hierarch", 'mensa']

n_features_list = np.concatenate(([1], np.arange(100, 1001, 100)))

if __name__ == "__main__":
    for n_features in n_features_list:
        data_config = load_config(cfg.DGP_CONFIGS_DIR, f"synthetic_se.yaml")
        data_config['n_samples'] = 20000
        data_config['n_features'] = n_features
        dl = SingleEventSyntheticDataLoader().load_data(data_config=data_config,
                                                        linear=True, copula_name=None,
                                                        k_tau=0, device=device, dtype=dtype)
        train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                          random_state=0)
        n_samples = train_dict['X'].shape[0]
        time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype).to(device)
        time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))
        
        for model_name in model_names:
            if model_name == "deepsurv":
                config = dotdict(cfg.DEEPSURV_PARAMS)
                model = DeepSurv(in_features=n_features, config=config)
            elif model_name == "deephit":
                config = dotdict(cfg.DEEPHIT_PARAMS)
                model = make_deephit_single(in_features=n_features, out_features=len(time_bins),
                                            time_bins=time_bins.cpu().numpy(), device=device, config=config)
            elif model_name == "mtlr":
                config = dotdict(cfg.MTLR_PARAMS)
                num_time_bins = len(time_bins)
                model = mtlr(in_features=n_features, num_time_bins=num_time_bins, config=config)
            elif model_name == "dsm":
                config = dotdict(cfg.DSM_PARAMS)
                n_iter = 1
                learning_rate = config['learning_rate']
                batch_size = config['batch_size']
                model = make_dsm_model(config)
                model.fit(train_dict['X'].cpu().numpy(), train_dict['T'].cpu().numpy(), train_dict['E'].cpu().numpy(),
                    val_data=(valid_dict['X'].cpu().numpy(), valid_dict['T'].cpu().numpy(), valid_dict['T'].cpu().numpy()),
                    learning_rate=learning_rate, batch_size=batch_size, iters=n_iter)
            elif model_name == "hierarch":
                config = load_config(cfg.HIERARCH_CONFIGS_DIR, f"synthetic_se.yaml")
                n_time_bins = len(time_bins)
                train_data, valid_data, test_data = format_hierarchical_data_cr(train_dict, valid_dict, test_dict,
                                                                                n_time_bins, 1, censoring_event=False)
                config['min_time'] = int(train_data[1].min())
                config['max_time'] = int(train_data[1].max())
                config['num_bins'] = len(time_bins)
                params = cfg.HIERARCH_PARAMS
                params['n_batches'] = int(n_samples/params['batch_size'])
                params['layer_size_fine_bins'] = [(100, 5)]
                layer_size = params['layer_size_fine_bins'][0][0]
                params['layer_size_fine_bins'] = calculate_layer_size_hierarch(layer_size, n_time_bins)
                hyperparams = format_hierarchical_hyperparams(params)
                verbose = params['verbose']
                model = util_torch.get_model_and_output("hierarch_full", train_data, test_data,
                                                        valid_data, config, hyperparams, verbose, model_only=True)
            elif model_name == "mensa":
                config = load_config(cfg.MENSA_CONFIGS_DIR, f"synthetic.yaml")
                n_dists = config['n_dists']
                layers = config['layers']
                dropout_rate = config['dropout_rate']
                model = MENSA(n_features, layers=layers, dropout_rate=dropout_rate,
                            n_events=1, n_dists=n_dists, device=device)
            else:
                raise NotImplementedError()

            # Calculate number of trainable parameters
            sum_params = 0
            if model_name == "deepsurv":
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                sum_params += total_params
            elif model_name == "deephit":
                total_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
                sum_params += total_params
            elif model_name == "mtlr":
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                sum_params += total_params
            elif model_name == "dsm":
                total_params = sum(p.numel() for p in model.torch_model.parameters() if p.requires_grad)
                sum_params += total_params
            elif model_name == "hierarch":
                sum_params += sum(p.numel() for p in model.main_layers[0].parameters() if p.requires_grad)
                for i in range(1):
                    sum_params += sum(p.numel() for p in model.event_networks[i].parameters() if p.requires_grad)
            elif model_name == "mensa":
                sum_params += sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            else:
                raise NotImplementedError()
                
            # Calculate number of FLOPs
            sum_flops = 0
            if model_name == "deepsurv":
                model.eval()
                flops = FlopCountAnalysis(model, test_dict['X'][0].unsqueeze(0).to(device))
                sum_flops += flops.total()
            elif model_name == "deephit":
                model.net.eval()
                flops = FlopCountAnalysis(model.net, test_dict['X'][0].unsqueeze(0).to(device))
                sum_flops += flops.total()
            elif model_name == "mtlr":
                model.eval()
                flops = FlopCountAnalysis(model, test_dict['X'][0].unsqueeze(0).to(device))
                sum_flops += flops.total()
            elif model_name == "dsm":
                flops = FlopCountAnalysis(model.torch_model, test_dict['X'][0].unsqueeze(0).to("cpu"))
                sum_flops += flops.total()
            elif model_name == "hierarch":
                model.eval()
                flops = FlopCountAnalysis(model, test_dict['X'][0].unsqueeze(0).to(device))
                sum_flops += flops.total()
            elif model_name == "mensa":
                model.model.eval()
                flops = FlopCountAnalysis(model.model, test_dict['X'][0].unsqueeze(0).to(device))
                sum_flops += flops.total()
            else:
                raise NotImplementedError()
            
            # Save results
            result_row = pd.Series([model_name, n_features, sum_params, sum_flops],
                                index=["ModelName", "NumFeatures", "SumParams", "SumFlops"])
            filename = f"{cfg.RESULTS_DIR}/complexcity_single_event_dgp.csv"
            if os.path.exists(filename):
                results = pd.read_csv(filename)
            else:
                results = pd.DataFrame(columns=result_row.keys())
            results = results.append(result_row, ignore_index=True)
            results.to_csv(filename, index=False)
        