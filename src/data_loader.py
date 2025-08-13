import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
import config as cfg
import numpy as np
from pycop import simulation
from utility.data import kendall_tau_to_theta
from utility.survival import make_stratified_split, make_multi_event_stratified_column
from dgp import DGP_Weibull_linear, DGP_Weibull_nonlinear
import torch
import random
from data import mimic_feature_selection 

class BaseDataLoader(ABC):
    """
    Base class for data loaders.
    """
    def __init__(self):
        """Initilizer method that takes a file path, file name,
        settings and optionally a converter"""
        self.X: pd.DataFrame = None
        self.y_t: List[np.ndarray] = None
        self.y_e: List[np.ndarray] = None
        self.num_features: List[str] = None
        self.cat_features: List[str] = None
        self.min_time = None
        self.max_time = None
        self.n_events = None
        self.params = None

    @abstractmethod
    def load_data(self, n_samples) -> None:
        """Loads the data from a data set at startup"""
        
    @abstractmethod
    def split_data(self) -> None:
        """Loads the data from a data set at startup"""

    def get_data(self) -> pd.DataFrame:
        """
        This method returns the features and targets
        :returns: X, y_t and y_e
        """
        return self.X, self.y_t, self.y_e

    def get_features(self) -> List[str]:
        """
        This method returns the names of numerical and categorial features
        :return: the columns of X as a list
        """
        return self.num_features, self.cat_features

    def _get_num_features(self, data) -> List[str]:
        return data.select_dtypes(include=np.number).columns.tolist()

    def _get_cat_features(self, data) -> List[str]:
        return data.select_dtypes(['object']).columns.tolist()


def get_data_loader(dataset_name: str) -> BaseDataLoader:
    if dataset_name in ["synthetic_se", "seer_se", "mimic_se", "proact_se"]:
        if dataset_name == "synthetic_se":
            return SingleEventSyntheticDataLoader()
        elif dataset_name == "seer_se":
            return SeerSingleDataLoader()
        elif dataset_name == "mimic_se":
            return MimicSingleDataLoader()
        elif dataset_name == "proact_se":
            return PROACTSingleDataLoader()
    elif dataset_name in ["synthetic_cr", "mimic_cr", "seer_cr", "rotterdam_cr"]:
        if dataset_name == "synthetic_cr":
            return CompetingRiskSyntheticDataLoader()
        elif dataset_name == "mimic_cr":
            return MimicCompetingDataLoader()
        elif dataset_name == "seer_cr":
            return SeerCompetingDataLoader()
        elif dataset_name == "rotterdam_cr":
            return RotterdamCompetingDataLoader()
    elif dataset_name in ["proact_me", "mimic_me", "synthetic_me", "ebmt_me", "rotterdam_me"]:
        if dataset_name == "proact_me":
            return PROACTMultiDataLoader()
        elif dataset_name == "mimic_me":
            return MimicMultiDataLoader()
        elif dataset_name == "synthetic_me":
            return MultiEventSyntheticDataLoader()
        elif dataset_name == "ebmt_me":
            return EBMTDataLoader()
        elif dataset_name == "rotterdam_me":
            return RotterdamMultiDataLoader()
    else:
        raise ValueError("Dataset not found")

class SingleEventSyntheticDataLoader(BaseDataLoader):
    def load_data(self, data_config, copula_name='clayton', k_tau=0,
                  linear=True, device='cpu', dtype=torch.float32):
        """
        This method generates synthetic data for single event (and censoring)
        DGP1: Data generation process for event
        DGP2: Data generation process for censoring
        """
        alpha_e1 = data_config['alpha_e1']
        alpha_e2 = data_config['alpha_e2']
        gamma_e1 = data_config['gamma_e1']
        gamma_e2 = data_config['gamma_e2']
        n_samples = data_config['n_samples']
        n_features = data_config['n_features']
        
        X = torch.rand((n_samples, n_features), device=device, dtype=dtype)

        if linear:
            dgp1 = DGP_Weibull_linear(n_features, alpha_e1, gamma_e1, device, dtype)
            dgp2 = DGP_Weibull_linear(n_features, alpha_e2, gamma_e2, device, dtype)
        else:
            dgp1 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e1,
                                         gamma=gamma_e1, device=device, dtype=dtype)
            dgp2 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e2,
                                         gamma=gamma_e2, device=device, dtype=dtype)
            
        if copula_name is None or k_tau == 0:
            rng = np.random.default_rng(0)
            u = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            v = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            uv = torch.stack([u, v], dim=1)
        else:
            theta = kendall_tau_to_theta(copula_name, k_tau)
            u, v = simulation.simu_archimedean(copula_name, 2, X.shape[0], theta=theta)
            u = torch.from_numpy(u).type(dtype).reshape(-1,1)
            v = torch.from_numpy(v).type(dtype).reshape(-1,1)
            uv = torch.cat([u, v], axis=1)
        
        t1_times = dgp1.rvs(X, uv[:,0].to(device))
        t2_times = dgp2.rvs(X, uv[:,1].to(device))
        
        observed_times = np.minimum(t1_times, t2_times)
        event_indicators = np.array((t2_times < t1_times), dtype=np.int32)
    
        columns = [f'X{i}' for i in range(n_features)]
        self.X = pd.DataFrame(X.cpu(), columns=columns)
        self.y_e = event_indicators
        self.y_t = observed_times
        self.dgps = [dgp1, dgp2]
        self.n_events = 1
        
        return self
    
    def split_data(self, train_size: float, valid_size: float,
                   test_size: float, dtype=torch.float32, random_state=0):
        df = pd.DataFrame(self.X)
        df['event'] = self.y_e
        df['time'] = self.y_t
    
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='time', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state)
    
        dataframes = [df_train, df_valid, df_test]
        dicts = []
        n_features = df.shape[1] - 2
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = torch.tensor(dataframe.loc[:, 'X0':f'X{n_features-1}'].to_numpy(), dtype=torch.float32)
            data_dict['E'] = torch.tensor(dataframe['event'].to_numpy(), dtype=torch.int32)
            data_dict['T'] = torch.tensor(dataframe['time'].to_numpy(), dtype=torch.float32)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]

class CompetingRiskSyntheticDataLoader(BaseDataLoader):
    def load_data(self, data_config, copula_name='clayton', k_tau=0,
                  linear=True, device='cpu', dtype=torch.float32):
        """
        This method generates synthetic data for 2 competing risks (and censoring)
        DGP1: Data generation process for event 1
        DGP2: Data generation process for event 2
        DGP3: Data generation process for censoring
        """
        alpha_e1 = data_config['alpha_e1']
        alpha_e2 = data_config['alpha_e2']
        alpha_e3 = data_config['alpha_e3']
        gamma_e1 = data_config['gamma_e1']
        gamma_e2 = data_config['gamma_e2']
        gamma_e3 = data_config['gamma_e3']
        n_samples = data_config['n_samples']
        n_features = data_config['n_features']
        
        X = torch.rand((n_samples, n_features), device=device, dtype=dtype)

        if linear:
            dgp1 = DGP_Weibull_linear(n_features, alpha_e1, gamma_e1, device, dtype)
            dgp2 = DGP_Weibull_linear(n_features, alpha_e2, gamma_e2, device, dtype)
            dgp3 = DGP_Weibull_linear(n_features, alpha_e3, gamma_e3, device, dtype)
        else:
            dgp1 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e1,
                                         gamma=gamma_e1, device=device, dtype=dtype)
            dgp2 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e2,
                                         gamma=gamma_e2, device=device, dtype=dtype)
            dgp3 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e3,
                                         gamma=gamma_e3, device=device, dtype=dtype)
        
        if copula_name is None or k_tau == 0:
            rng = np.random.default_rng(0)
            u = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            v = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            w = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            uvw = torch.stack([u, v, w], dim=1)
        else:
            theta = kendall_tau_to_theta(copula_name, k_tau)
            u, v, w = simulation.simu_archimedean(copula_name, 3, X.shape[0], theta=theta)
            u = torch.from_numpy(u).type(dtype).reshape(-1,1)
            v = torch.from_numpy(v).type(dtype).reshape(-1,1)
            w = torch.from_numpy(w).type(dtype).reshape(-1,1)
            uvw = torch.cat([u,v,w], axis=1).to(device)
            
        t1_times = dgp1.rvs(X, uvw[:,0].to(device))
        t2_times = dgp2.rvs(X, uvw[:,1].to(device))
        t3_times = dgp2.rvs(X, uvw[:,2].to(device))
        
        event_times = np.concatenate([t1_times.reshape(-1,1),
                                      t2_times.reshape(-1,1),
                                      t3_times.reshape(-1,1)], axis=1)
        event_indicators = np.argmin(event_times, axis=1)
        observed_times = event_times[np.arange(event_times.shape[0]), event_indicators]
        
        columns = [f'X{i}' for i in range(n_features)]
        self.X = pd.DataFrame(X.cpu(), columns=columns)
        self.y_e = event_indicators
        self.y_t = observed_times
        self.y_t1 = t1_times
        self.y_t2 = t2_times
        self.y_t3 = t3_times
        
        self.n_events = 2
        self.dgps = [dgp1, dgp2, dgp3]
        
        return self
   
    def split_data(self,
                   train_size: float,
                   valid_size: float,
                   test_size: float,
                   dtype=torch.float32,
                   random_state=0):
        df = pd.DataFrame(self.X)
        df['event'] = self.y_e
        df['time'] = self.y_t
        df['t1'] = self.y_t1
        df['t2'] = self.y_t2
        df['t3'] = self.y_t3
        
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='time', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state)
        
        dataframes = [df_train, df_valid, df_test]
        dicts = []
        n_features = df.shape[1] - 5
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = torch.tensor(dataframe.loc[:, 'X0':f'X{n_features-1}'].to_numpy(), dtype=torch.float32)
            data_dict['E'] = torch.tensor(dataframe['event'].to_numpy(), dtype=torch.int32)
            data_dict['T'] = torch.tensor(dataframe['time'].to_numpy(), dtype=torch.float32)
            data_dict['T1'] = torch.tensor(dataframe['t1'].to_numpy(), dtype=torch.float32)
            data_dict['T2'] = torch.tensor(dataframe['t2'].to_numpy(), dtype=torch.float32)
            data_dict['T3'] = torch.tensor(dataframe['t3'].to_numpy(), dtype=torch.float32)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]
        
class MultiEventSyntheticDataLoader(BaseDataLoader):
    def load_data(self, data_config, copula_names=["clayton", "clayton", "clayton"],
                  k_taus=[0, 0, 0], linear=True, device='cpu', dtype=torch.float32):
        """
        This method generates synthetic data for 3 multiple events (with adm. censoring)
        DGP1: Data generation process for event 1
        DGP2: Data generation process for event 2
        DGP3: Data generation process for event 3
        """
        alpha_e1 = data_config['alpha_e1']
        alpha_e2 = data_config['alpha_e2']
        alpha_e3 = data_config['alpha_e3']
        alpha_e4 = data_config['alpha_e4']
        gamma_e1 = data_config['gamma_e1']
        gamma_e2 = data_config['gamma_e2']
        gamma_e3 = data_config['gamma_e3']
        gamma_e4 = data_config['gamma_e4']
        n_samples = data_config['n_samples']
        n_features = data_config['n_features']
        adm_censoring_time = data_config['adm_censoring_time']
        
        X = torch.rand((n_samples, n_features), device=device, dtype=dtype)
        
        if linear:
            dgp1 = DGP_Weibull_linear(n_features, alpha_e1, gamma_e1, device, dtype)
            dgp2 = DGP_Weibull_linear(n_features, alpha_e2, gamma_e2, device, dtype)
            dgp3 = DGP_Weibull_linear(n_features, alpha_e3, gamma_e3, device, dtype)
            dgp4 = DGP_Weibull_linear(n_features, alpha_e4, gamma_e4, device, dtype)
        else:
            dgp1 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e1,
                                         gamma=gamma_e1, device=device, dtype=dtype)
            dgp2 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e2,
                                         gamma=gamma_e2, device=device, dtype=dtype)
            dgp3 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e3,
                                         gamma=gamma_e3, device=device, dtype=dtype)
            dgp4 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e4,
                                         gamma=gamma_e4, device=device, dtype=dtype)

        if copula_names is None or all(v == 0 for v in k_taus):
            rng = np.random.default_rng(0)
            u = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            v = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            w = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            z = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            uvwz = torch.stack([u, v, w, z], dim=1)
        else:
            raise NotImplementedError()
        
        t1_times = dgp1.rvs(X, uvwz[:,0].to(device))
        t2_times = dgp2.rvs(X, uvwz[:,1].to(device))
        t3_times = dgp2.rvs(X, uvwz[:,2].to(device))
        t4_times = dgp2.rvs(X, uvwz[:,3].to(device))
        
        # Make adm. censoring
        event_times = np.stack([t1_times, t2_times, t3_times, t4_times], axis=1)
        event_times = np.minimum(event_times, adm_censoring_time)
        event_indicators = (event_times < adm_censoring_time).astype(int)

        # Format data
        columns = [f'X{i}' for i in range(n_features)]
        self.X = pd.DataFrame(X.cpu(), columns=columns)
        self.y_t = event_times
        self.y_e = event_indicators
        self.dgps = [dgp1, dgp2, dgp3, dgp4]
        self.n_events = 4
        
        return self
    
    def split_data(self, train_size: float, valid_size: float,
                   test_size: float, dtype=torch.float32, random_state=0):
        df = pd.DataFrame(self.X)
        df['e1'] = self.y_e[:,0]
        df['e2'] = self.y_e[:,1]
        df['e3'] = self.y_e[:,2]
        df['e4'] = self.y_e[:,3]
        df['t1'] = self.y_t[:,0]
        df['t2'] = self.y_t[:,1]
        df['t3'] = self.y_t[:,2]
        df['t4'] = self.y_t[:,3]
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='multi', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size, n_events=4,
                                                            random_state=random_state)
        
        dataframes = [df_train, df_valid, df_test]
        dicts = []
        n_features = df.shape[1] - 8
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = torch.tensor(dataframe.loc[:, 'X0':f'X{n_features-1}'].to_numpy(), dtype=torch.float32)
            data_dict['E'] = torch.stack([torch.tensor(dataframe['e1'].values, dtype=torch.int32),
                                          torch.tensor(dataframe['e2'].values, dtype=torch.int32),
                                          torch.tensor(dataframe['e3'].values, dtype=torch.int32),
                                          torch.tensor(dataframe['e4'].values, dtype=torch.int32)], axis=1)
            data_dict['T'] = torch.stack([torch.tensor(dataframe['t1'].values, dtype=torch.float32),
                                          torch.tensor(dataframe['t2'].values, dtype=torch.float32),
                                          torch.tensor(dataframe['t3'].values, dtype=torch.float32),
                                          torch.tensor(dataframe['t4'].values, dtype=torch.float32)], axis=1)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]

class PROACTSingleDataLoader(BaseDataLoader):
    """
    Data loader for ALS dataset (ME). Use the PRO-ACT dataset.
    """
    def load_data(self, n_samples:int = None):
        df = pd.read_csv(f'{cfg.DATA_DIR}/proact_processed.csv', index_col=0)
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
        label_cols = [col for col in df.columns if any(substring in col for substring in ['Event', 'TTE'])]
        event_names = ['Speech']
        for event_name in event_names:
            df = df.loc[(df[f'Event_{event_name}'] == 0) | (df[f'Event_{event_name}'] == 1)] # drop already occured
            df = df.loc[(df[f'TTE_{event_name}'] > 0) & (df[f'TTE_{event_name}'] <= 500)] # 1 - 500
        df = df.drop(df.filter(like='_Strength').columns, axis=1) # Drop strength tests
        df = df.drop('Race_Caucasian', axis=1) # Drop race information
        df = df.drop('El_escorial', axis=1) # Drop el_escorial
        df = df.drop(['Height', 'Weight', 'BMI'], axis=1) # Drop height/weight/bmi
        self.X = df.drop(label_cols, axis=1)
        self.columns = list(self.X.columns)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        times = [df[f'TTE_{event_col}'].values for event_col in event_names]
        events = [df[f'Event_{event_col}'].values for event_col in event_names]
        self.y_t = times[0]
        self.y_e = events[0]
        self.n_events = 1
        self.trajectories = []
        return self

    def split_data(self, train_size: float, valid_size: float,
                   test_size: float, dtype=torch.float32, random_state=0):
        df = pd.DataFrame(self.X)
        df['event'] = self.y_e
        df['time'] = self.y_t
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='time', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state)
        
        dataframes = [df_train, df_valid, df_test]
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = dataframe.drop(['event', 'time'], axis=1).to_numpy()
            data_dict['E'] = torch.tensor(dataframe['event'].to_numpy(dtype=np.int32), dtype=torch.int32)
            data_dict['T'] = torch.tensor(dataframe['time'].to_numpy(dtype=np.float32), dtype=torch.float32)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]

class PROACTMultiDataLoader(BaseDataLoader):
    """
    Data loader for ALS dataset (ME). Use the PRO-ACT dataset.
    """
    def load_data(self, n_samples:int = None):
        df = pd.read_csv(f'{cfg.DATA_DIR}/proact_processed.csv', index_col=0)
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
        label_cols = [col for col in df.columns if any(substring in col for substring in ['Event', 'TTE'])]
        event_names = ['Speech', 'Swallowing', 'Handwriting', 'Walking']
        for event_name in event_names:
            df = df.loc[(df[f'Event_{event_name}'] == 0) | (df[f'Event_{event_name}'] == 1)] # drop already occured
            df = df.loc[(df[f'TTE_{event_name}'] > 0) & (df[f'TTE_{event_name}'] <= 500)] # 1 - 500
        df = df.drop(df.filter(like='_Strength').columns, axis=1) # Drop strength tests
        df = df.drop('Race_Caucasian', axis=1) # Drop race information
        df = df.drop('El_escorial', axis=1) # Drop el_escorial
        df = df.drop(['Height', 'Weight', 'BMI'], axis=1) # Drop height/weight/bmi
        self.X = df.drop(label_cols, axis=1)
        self.columns = list(self.X.columns)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        times = [df[f'TTE_{event_col}'].values for event_col in event_names]
        events = [df[f'Event_{event_col}'].values for event_col in event_names]
        self.y_t = np.stack((times[0], times[1], times[2], times[3]), axis=1)
        self.y_e = np.stack((events[0], events[1], events[2], events[3]), axis=1)
        self.n_events = 4
        self.trajectories = []
        return self

    def split_data(self, train_size: float, valid_size: float,
                   test_size: float, dtype=torch.float32, random_state=0):
        df = pd.DataFrame(self.X)
        df['e1'] = self.y_e[:,0]
        df['e2'] = self.y_e[:,1]
        df['e3'] = self.y_e[:,2]
        df['e4'] = self.y_e[:,3]
        df['t1'] = self.y_t[:,0]
        df['t2'] = self.y_t[:,1]
        df['t3'] = self.y_t[:,2]
        df['t4'] = self.y_t[:,3]
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='multi', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state, n_events=4)
        
        dataframes = [df_train, df_valid, df_test]
        event_cols = ['e1', 'e2', 'e3', 'e4']
        time_cols = ['t1', 't2', 't3', 't4']
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = dataframe.drop(event_cols + time_cols, axis=1).values
            data_dict['E'] = np.stack([dataframe['e1'].values, dataframe['e2'].values,
                                       dataframe['e3'].values, dataframe['e4'].values], axis=1).astype(np.int32)
            data_dict['T'] = np.stack([dataframe['t1'].values, dataframe['t2'].values,
                                       dataframe['t3'].values, dataframe['t4'].values], axis=1).astype(np.float32)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]

class MimicMultiDataLoader(BaseDataLoader):
    """
    Data loader for MIMIC dataset (ME)
    """
    def load_data(self, n_samples:int = None):
        '''
        t and e order, followed by arf, shock, death
        '''
        df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, 'mimic.csv.gz'), compression='gzip', index_col=0)
        df = df[mimic_feature_selection.selected_features]

        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
            
        df = df[(df['Age'] >= 60) & (df['Age'] <= 65)] # select cohort ages 60-65    
        
        df = df[df['ARF_time'] > 0]
        df = df[df['shock_time'] > 0]
        df = df[df['death_time'] > 0]
        
        columns_to_drop = [col for col in df.columns if
                           any(substring in col for substring in ['_event', '_time', 'hadm_id'])]
        events = ['ARF', 'shock', 'death']
        self.X = df.drop(columns_to_drop, axis=1)
        self.columns = list(self.X.columns)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        
        times = [df[f'{event_col}_time'].values for event_col in events]
        events = [df[f'{event_col}_event'].values for event_col in events]
        self.y_t = np.stack((times[0], times[1], times[2]), axis=1)
        self.y_e = np.stack((events[0], events[1], events[2]), axis=1)
        self.n_events = 3
        self.trajectories = []
        
        return self

    def split_data(self, train_size: float, valid_size: float,
                   test_size: float, dtype=torch.float32, random_state=0):
        df = pd.DataFrame(self.X)
        df['e1'] = self.y_e[:,0]
        df['e2'] = self.y_e[:,1]
        df['e3'] = self.y_e[:,2]
        df['t1'] = self.y_t[:,0]
        df['t2'] = self.y_t[:,1]
        df['t3'] = self.y_t[:,2]
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='multi', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size, n_events=3,
                                                            random_state=random_state)
        
        dataframes = [df_train, df_valid, df_test]
        event_cols = ['e1', 'e2', 'e3']
        time_cols = ['t1', 't2', 't3']
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = dataframe.drop(event_cols + time_cols, axis=1).values
            data_dict['E'] = np.stack([dataframe['e1'].values, dataframe['e2'].values,
                                       dataframe['e3'].values], axis=1).astype(np.int32)
            data_dict['T'] = np.stack([dataframe['t1'].values, dataframe['t2'].values,
                                       dataframe['t3'].values], axis=1).astype(np.float32)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]

class SeerSingleDataLoader(BaseDataLoader):
    """
    Data loader for SEER dataset (SE)
    """
    def load_data(self, n_samples:int = None):
        df = pd.read_csv(f'{cfg.DATA_DIR}/seer_processed.csv')
        
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
            
        # Select cohort of newly-diagnosed patients
        df = df.loc[df['Year of diagnosis'] == 0]
        df = df.drop('Year of diagnosis', axis=1)
            
        self.X = df.drop(['duration', 'event_heart', 'event_breast'], axis=1)
        self.columns = list(self.X.columns)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)

        encoded_events = np.zeros(len(df), dtype=int)
        encoded_events[df['event_breast'] == 1] = 1
        encoded_events[df['event_heart'] == 1] = 0 # event is censored

        self.y_t = np.array(df['duration'])
        self.y_e = encoded_events
        self.n_events = 1
        
        return self
    
    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float32,
                random_state=0):
        df = pd.DataFrame(self.X)
        df['event'] = self.y_e
        df['time'] = self.y_t
        
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='time', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state)
        
        dataframes = [df_train, df_valid, df_test]
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = dataframe.drop(['event', 'time'], axis=1).to_numpy()
            data_dict['E'] = torch.tensor(dataframe['event'].to_numpy(dtype=np.int32), dtype=torch.int32)
            data_dict['T'] = torch.tensor(dataframe['time'].to_numpy(dtype=np.float32), dtype=torch.float32)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]
    
class MimicSingleDataLoader(BaseDataLoader):
    """
    Data loader for MIMIC dataset (SE)
    """
    def load_data(self, n_samples:int = None):
        '''
        t and e order, followed by death
        '''
        df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, 'mimic.csv.gz'), compression='gzip', index_col=0)
        df = df[mimic_feature_selection.selected_features]
        
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
            
        df = df[(df['Age'] >= 60) & (df['Age'] <= 65)] # select cohort ages 60-65
        
        df = df[df['ARF_time'] > 0]
        df = df[df['shock_time'] > 0]
        df = df[df['death_time'] > 0]
  
        columns_to_drop = [col for col in df.columns if
                           any(substring in col for substring in ['_event', '_time', 'hadm_id'])]
        events = ['death']
        self.X = df.drop(columns_to_drop, axis=1)
        self.columns = list(self.X.columns)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        
        self.y_t = df[f'death_time'].values # use only death
        self.y_e = df[f'death_event'].values
        self.n_events = 1
        
        return self

    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float32,
                random_state=0):
        df = pd.DataFrame(self.X)
        df['event'] = self.y_e
        df['time'] = self.y_t
        
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='time', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state)
        
        dataframes = [df_train, df_valid, df_test]
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = dataframe.drop(['event', 'time'], axis=1).to_numpy()
            data_dict['E'] = torch.tensor(dataframe['event'].to_numpy(dtype=np.int32), dtype=torch.int32)
            data_dict['T'] = torch.tensor(dataframe['time'].to_numpy(dtype=np.float32), dtype=torch.float32)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]
    
class SeerCompetingDataLoader(BaseDataLoader):
    """
    Data loader for SEER dataset (CR)
    """
    def load_data(self, n_samples:int = None):
        df = pd.read_csv(f'{cfg.DATA_DIR}/seer_processed.csv')
        
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)

        # Select cohort of newly-diagnosed patients
        df = df.loc[df['Year of diagnosis'] == 0]
        df = df.drop('Year of diagnosis', axis=1)
            
        self.X = df.drop(['duration', 'event_heart', 'event_breast'], axis=1)
        self.columns = list(self.X.columns)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)

        encoded_events = np.zeros(len(df), dtype=int)
        encoded_events[df['event_heart'] == 1] = 1
        encoded_events[df['event_breast'] == 1] = 2

        self.y_t = np.array(df['duration'])
        self.y_e = encoded_events
        self.n_events = 2
        
        return self
    
    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float32,
                random_state=0):
        df = pd.DataFrame(self.X)
        df['event'] = self.y_e
        df['time'] = self.y_t
        
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='time', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state)
        
        dataframes = [df_train, df_valid, df_test]
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = dataframe.drop(['event', 'time'], axis=1).to_numpy()
            data_dict['E'] = torch.tensor(dataframe['event'].to_numpy(dtype=np.int32), dtype=torch.int32)
            data_dict['T'] = torch.tensor(dataframe['time'].to_numpy(dtype=np.float32), dtype=torch.float32)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]
        
class RotterdamCompetingDataLoader(BaseDataLoader):
    """
    Data loader for Rotterdam dataset (CR)
    """
    def load_data(self, n_samples:int = None):
        '''
        Events: 0 censor, 1 death, 2 recur
        '''
        df = pd.read_csv(f'{cfg.DATA_DIR}/rotterdam.csv')
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
        size_mapping = {
            '<=20': 10,
            '20-50': 35,
            '>50': 75
        }
        # Apply mapping
        df['size_map'] = df['size'].replace(size_mapping)
        
        def get_time(row):
            if row['event'] == 0:
                return min(row['rtime'], row['dtime'])
            elif row['event'] == 1:
                return row['dtime']
            elif row['event'] == 2:
                return row['rtime']
            else:
                raise ValueError("error in time")
 
        def get_event(row):
            if row['recur'] == 0 and row['death'] == 0:
                return 0
            elif row['rtime'] <= row['dtime'] and row['recur'] == 1:
                return 2
            elif row['dtime'] <= row['rtime'] and row['death'] == 1:
                return 1
            elif row['death'] == 1 and row['recur'] == 0: #some scenaro, recur time censor but earlier than death.
                return 1
            else:
                raise ValueError("error in event")
        
        df['event'] = df.apply(get_event, axis=1)
        df['time'] = df.apply(get_time, axis=1)
        self.X = df.drop(['pid', 'size', 'rtime', 'recur', 'dtime', 'death', 'time', 'event'], axis=1)
        self.columns = list(self.X.columns)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        self.y_t = df['time']
        self.y_e = df['event']
        self.n_events = 2
        return self
        
    def split_data(self, train_size: float, valid_size: float,
                   test_size: float, dtype=torch.float32, random_state=0):
        df = pd.DataFrame(self.X)
        df['event'] = self.y_e
        df['time'] = self.y_t
        
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='time', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state)
        
        dataframes = [df_train, df_valid, df_test]
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = dataframe.drop(['event', 'time'], axis=1).to_numpy()
            data_dict['E'] = torch.tensor(dataframe['event'].to_numpy(dtype=np.int32), dtype=torch.int32)
            data_dict['T'] = torch.tensor(dataframe['time'].to_numpy(dtype=np.float32), dtype=torch.float32)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]

class MimicCompetingDataLoader(BaseDataLoader):
    """
    Data loader for MIMIC dataset (CR)
    """
    def load_data(self, n_samples:int = None):
        '''
        t and e order, followed by death
        '''
        df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, 'mimic.csv.gz'), compression='gzip', index_col=0)
        df = df[mimic_feature_selection.selected_features]
        
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
            
        df = df[(df['Age'] >= 60) & (df['Age'] <= 65)] # select cohort ages 60-65
        
        df = df[df['ARF_time'] > 0]
        df = df[df['shock_time'] > 0]
        df = df[df['death_time'] > 0]
  
        columns_to_drop = [col for col in df.columns if
                           any(substring in col for substring in ['_event', '_time', 'hadm_id'])]
        events = ['death']
        self.X = df.drop(columns_to_drop, axis=1)
        self.columns = list(self.X.columns)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)

        def get_time(row):
            if row['event'] == 0:
                return max([row['ARF_time'], row['death_time'], row['shock_time']])
            elif row['event'] == 3:
                return row['death_time']
            elif row['event'] == 2:
                return row['shock_time']
            elif row['event'] == 1:
                return row['ARF_time']        
            else:
                raise ValueError("error in time")
        
        def get_event(row):
            if row['ARF_event'] == 0 and row['shock_event'] == 0 and row['death_event'] == 0:
                return 0
            elif row['death_event'] == 1 and min([row['ARF_time'], row['death_time'], row['shock_time']]) == row['death_time']:
                return 3
            elif row['shock_event'] == 1 and min([row['ARF_time'], row['shock_time']]) == row['shock_time']:
                return 2
            elif row['ARF_event'] == 1:
                return 1
            else:
                print (row)
                raise ValueError("error in event")
        
        df['event'] = df.apply(get_event, axis=1)
        df['time'] = df.apply(get_time, axis=1)
        
        self.y_t = df[f'time'].values 
        self.y_e = df[f'event'].values # 0 (censored), 1 ARF, 2 shock, 3 death
        self.n_events = 3
        
        return self

    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float32,
                random_state=0):
        df = pd.DataFrame(self.X)
        df['event'] = self.y_e
        df['time'] = self.y_t
        
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='time', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state)
        
        dataframes = [df_train, df_valid, df_test]
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = dataframe.drop(['event', 'time'], axis=1).to_numpy()
            data_dict['E'] = torch.tensor(dataframe['event'].to_numpy(dtype=np.int32), dtype=torch.int32)
            data_dict['T'] = torch.tensor(dataframe['time'].to_numpy(dtype=np.float32), dtype=torch.float32)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]

class EBMTDataLoader(BaseDataLoader):
    """
    Data loader for EBMT dataset (ME)
    """
    def load_data(self, n_samples: int = None):
        """
        t and e order, followed by arf, shock, death
        """
        df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, 'ebmt.csv'), index_col=0)

        if n_samples:
            df = df.sample(n=n_samples, random_state=0)

        # Define columns
        self.events_names = ['2', '3', '4', '5', '6'] 
        self.X_columns = ['match_no gender mismatch', 'proph_yes', 'year_1990-1994', 'year_1995-1998', 'agecl_<=20', 'agecl_>40']
        self.E_columns = ['e1', 'e2', 'e3', 'e4', 'e5']
        self.T_columns = ['t1', 't2', 't3', 't4', 't5']
        self.columns = list(self.X_columns)
        
        # Rename time and event columns
        df = df.rename(columns={
            '2_time': 't1', '3_time': 't2', '4_time': 't3', '5_time': 't4', '6_time': 't5',
            '2_event': 'e1', '3_event': 'e2', '4_event': 'e3', '5_event': 'e4', '6_event': 'e5'
        })

        # Remove rows where any duration ≤ 0
        df = df[(df[self.T_columns] > 0).all(axis=1)]

        # Save original df
        self.df = df

        # Features
        self.X = df[self.X_columns]
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)

        # Targets
        self.y_t = df[self.T_columns].to_numpy()
        self.y_e = df[self.E_columns].to_numpy()

        self.n_events = 5
        self.trajectories = [
            (1, 3), (2, 3),
            # (3, 4),
            (1, 5), (2, 5), (3, 5), (4, 5),
        ]

        return self

    def split_data(self, train_size: float, valid_size: float,
                   test_size: float, dtype=torch.float32, random_state=0):
        df_train, df_valid, df_test = make_stratified_split(self.df, stratify_colname='multi', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            n_events=self.n_events, random_state=random_state)
        
        dataframes = [df_train, df_valid, df_test]
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = dataframe[self.X_columns].values
            data_dict['E'] = dataframe[self.E_columns].astype(np.int32).values
            data_dict['T'] = dataframe[self.T_columns].astype(np.float32).values
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2] 

class RotterdamMultiDataLoader(BaseDataLoader):
    """
    Data loader for Rotterdam dataset (ME)
    """
    def load_data(self, n_samples:int = None):
        '''
        Events: 0 censor, 1 recur, 2 death
        '''
        df = pd.read_csv(f'{cfg.DATA_DIR}/rotterdam.csv')
        
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
        size_mapping = {
            '<=20': 10,
            '20-50': 35,
            '>50': 75
        }
        # Apply mapping
        df['size_map'] = df['size'].replace(size_mapping)
        self.df = df
        self.df = self.df.rename(columns={'rtime': 't1', 'dtime': 't2', 'recur': 'e1', 'death': 'e2'})
        self.X = df.drop(['pid', 'size', 'rtime', 'recur', 'dtime', 'death'], axis=1)
        self.columns = list(self.X.columns)
        self.X_columns = self.X.columns
        self.E_columns = ['e1', 'e2']
        self.T_columns = ['t1', 't2']
        self.y_t = np.stack((df['rtime'], df['dtime']), axis=1)
        self.y_e = np.stack((df['recur'], df['death']), axis=1)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        self.n_events = 2
        self.trajectories = [(1, 2)]
        return self
        
    def split_data(self, train_size: float, valid_size: float,
                   test_size: float, dtype=torch.float32, random_state=0):
        df_train, df_valid, df_test = make_stratified_split(self.df, stratify_colname='multi', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            n_events=self.n_events, random_state=random_state)
        
        dataframes = [df_train, df_valid, df_test]
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = dataframe[self.X_columns].values
            data_dict['E'] = dataframe[self.E_columns].astype(np.int32).values
            data_dict['T'] = dataframe[self.T_columns].astype(np.float32).values
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]       
