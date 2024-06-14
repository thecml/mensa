import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from typing import Tuple, List
from pathlib import Path
import config as cfg
from utility.survival import convert_to_structured
from utility.data import make_synthetic
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from utility.survival import get_trajectory_labels
from hierarchical.data_settings import *
import pickle
from pycop import simulation
from scipy import stats
from utility.data import (inverse_transform, inverse_transform_weibull, relu,
                          inverse_transform_exp, inverse_transform_lognormal)

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

class LinearSyntheticDataLoader(BaseDataLoader):
    def load_data(self, copula_name='Frank', theta=0.01,
                  n_samples=30000, n_features=10,
                  rng=np.random.default_rng(0)):
        # Generate synthetic data (time-to-event and censoring indicator)
        # with linear parametric proportional hazards model (Weibull CoxPH)
        v_e=4; rho_e=14; v_c=3; rho_c=16
        
        X = rng.uniform(0, 1, (n_samples, n_features))
        beta_c = rng.uniform(0, 1, (n_features,))
        beta_e = rng.uniform(0, 1, (n_features,))

        event_risk = np.matmul(X, beta_e).squeeze()
        censoring_risk = np.matmul(X, beta_c).squeeze()

        if copula_name=='Frank':
            u_e, u_c = simulation.simu_archimedean('frank', 2, n_samples, theta)
        elif copula_name=='Gumbel':
            u_e, u_c = simulation.simu_archimedean('gumbel', 2, n_samples, theta)
        elif copula_name=='Clayton':
            u_e, u_c = simulation.simu_archimedean('clayton', 2, n_samples, theta)
        elif copula_name=="Independent":
            u_e = rng.uniform(0, 1, n_samples)
            u_c = rng.uniform(0, 1, n_samples)
        else:
            raise ValueError('Copula not implemented') 

        event_time = inverse_transform(u_e, event_risk, v_e, rho_e)
        censoring_time = inverse_transform(u_c, censoring_risk, v_c, rho_c)

        observed_time = np.minimum(event_time, censoring_time)
        event_indicator = (event_time < censoring_time).astype(int)

        data = np.concatenate([X, observed_time[:, None], event_indicator[:, None],
                               event_time[:, None], censoring_time[:, None]], axis=1)
        label_cols = ['observed_time', 'event_indicator', 'event_time', 'censoring_time']
        columns = [f'X{i}' for i in range(n_features)] + label_cols
        df = pd.DataFrame(data, columns=columns)
        
        self.X = df.drop(label_cols, axis=1)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        self.y_t = df['observed_time'].values
        self.y_e = df['event_indicator'].values
        self.params = [beta_e, beta_c]
        return self
    
    def split_data(self,
                   train_size: float,
                   valid_size: float,
                   random_state=0):
        X = self.X
        y = convert_to_structured(self.y_t, self.y_e)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size,
                                                            random_state=0)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                              test_size=valid_size,
                                                              random_state=0)
        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

class NonlinearSyntheticDataLoader(BaseDataLoader):
    def load_data(self, copula_name='Frank', theta=0.1, n_samples=30000,
                  n_features=10, rng=np.random.default_rng(0)):
        # Generate synthetic data with parametric model (Weibull)
        hidden_dim = int(n_features/2)
        X = rng.uniform(0, 1, (n_samples, n_features))
        beta = rng.uniform(0, 1, (n_features, hidden_dim))
    
        beta_shape_c = rng.uniform(0, 1, (int(hidden_dim*0.8),))
        beta_scale_c = rng.uniform(0, 1, (int(hidden_dim*0.8),))
        
        beta_shape_e = rng.uniform(0, 1, (int(hidden_dim*0.8),))
        beta_scale_e = rng.uniform(0, 1, (int(hidden_dim*0.8),))

        beta_shape_e = np.pad(beta_shape_e, (0, 1)) # pad 0
        beta_scale_e = np.pad(beta_scale_e, (1, 0))
        beta_shape_c = np.pad(beta_shape_c, (0, 1))
        beta_scale_c = np.pad(beta_scale_c, (1, 0))
        
        hidden_rep = np.matmul(X, beta).squeeze()
        hidden_rep = relu(hidden_rep)
        shape_c = np.matmul(hidden_rep, beta_shape_c).squeeze()
        scale_c = np.matmul(hidden_rep, beta_scale_c).squeeze()
        shape_e = np.matmul(hidden_rep, beta_shape_e).squeeze()
        scale_e = np.matmul(hidden_rep, beta_scale_e).squeeze()

        if copula_name=='Frank':
            u_e, u_c = simulation.simu_archimedean('frank', 2, n_samples, theta)
        elif copula_name=='Gumbel':
            u_e, u_c = simulation.simu_archimedean('gumbel', 2, n_samples, theta)
        elif copula_name=='Clayton':
            u_e, u_c = simulation.simu_archimedean('clayton', 2, n_samples, theta)
        elif copula_name=="Independent":
            u_e = rng.uniform(0, 1, n_samples)
            u_c = rng.uniform(0, 1, n_samples)
        else:
            raise ValueError('Copula not implemented')
    
        event_time = inverse_transform_weibull(u_e, shape_e, scale_e)
        censoring_time = inverse_transform_weibull(u_c, shape_c, scale_c)
    
        observed_time = np.minimum(event_time, censoring_time)
        event_indicator = (event_time < censoring_time).astype(int)
        
        data = np.concatenate([X, observed_time[:, None], event_indicator[:, None], event_time[:, None],
                           censoring_time[:, None]], axis=1)
        label_cols = ['observed_time', 'event_indicator', 'event_time', 'censoring_time']
        columns = [f'X{i}' for i in range(n_features)] + label_cols
        df = pd.DataFrame(data, columns=columns)
    
        self.X = df.drop(label_cols, axis=1)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        self.y_t = df['observed_time'].values
        self.y_e = df['event_indicator'].values
        self.params = [beta, beta_shape_e, beta_scale_e]
        return self
    
    def split_data(self,
                   train_size: float,
                   valid_size: float,
                   random_state=0):
        X = self.X
        y = convert_to_structured(self.y_t, self.y_e)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size,
                                                            random_state=0)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                              test_size=valid_size,
                                                              random_state=0)
        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
    
class CompetingRiskSyntheticDataLoader(BaseDataLoader):
    def load_data(self, copula_parameters=None, dist='Weibull',
                  n_samples=30000, n_features=10, rng=np.random.default_rng(0)):
        # Generate synthetic data (2 competing risks and censoring)
        if copula_parameters is None:
            corrMatrix = np.array([[1, 0.8, 0], [0.8, 1, 0], [0, 0, 1]])
            copula_parameters = [
                {"type": "clayton", "weight": 1 / 3, "theta": 2},
                {"type": "frank", "weight": 1 / 3, "theta": 1},
                {"type": "gumbel", "weight": 1 / 3, "theta": 3}
            ]
        
        X = rng.uniform(0, 1, (n_samples, n_features))
        hidden_dim = int(n_features/2)
        beta = rng.uniform(0, 1, (n_features, hidden_dim))

        beta_shape_e1 = rng.uniform(0, 1, (int(hidden_dim*0.8),))
        beta_scale_e1 = rng.uniform(0, 1, (int(hidden_dim*0.8),))
        beta_shape_e2 = rng.uniform(0, 1, (int(hidden_dim*0.8),))
        beta_scale_e2 = rng.uniform(0, 1, (int(hidden_dim*0.8),))
        beta_shape_c = rng.uniform(0, 1, (int(hidden_dim*0.8),))
        beta_scale_c = rng.uniform(0, 1, (int(hidden_dim*0.8),))
        
        hidden_rep = np.matmul(X, beta).squeeze()
        hidden_rep = relu(hidden_rep)

        shape_e1 = np.matmul(hidden_rep[:, 0:int(hidden_dim*0.8)], beta_shape_e1).squeeze()
        scale_e1 = np.matmul(hidden_rep[:, -int(hidden_dim*0.8):], beta_scale_e1).squeeze()
        shape_e2 = np.matmul(hidden_rep[:, 0:int(hidden_dim*0.8)], beta_shape_e2).squeeze()
        scale_e2 = np.matmul(hidden_rep[:, -int(hidden_dim*0.8):], beta_scale_e2).squeeze()
        shape_c = np.matmul(hidden_rep[:, 0:int(hidden_dim*0.8)], beta_shape_c).squeeze()
        scale_c = np.matmul(hidden_rep[:, -int(hidden_dim*0.8):], beta_scale_c).squeeze()

        u_e1, u_e2, u_c = simulation.simu_mixture(3, n_samples, copula_parameters)

        if dist == "Weibull":
            e1_time = inverse_transform_weibull(u_e1, shape_e1, scale_e1)
            e2_time = inverse_transform_weibull(u_e2, shape_e2, scale_e2)
            c_time = inverse_transform_weibull(u_c, shape_c, scale_c)
        elif dist == "Exp":
            e1_time = inverse_transform_exp(u_e1, shape_e1, scale_e1)
            e2_time = inverse_transform_exp(u_e2, shape_e2, scale_e2)
            c_time = inverse_transform_exp(u_c, shape_c, scale_c)
        elif dist == "Lognormal":
            e1_time = inverse_transform_lognormal(u_e1, shape_e1, scale_e1)
            e2_time = inverse_transform_lognormal(u_e2, shape_e2, scale_e2)
            c_time = inverse_transform_lognormal(u_c, shape_c, scale_c)
        else:
            raise ValueError('Dist not implemented')
        
        all_times = np.stack([e1_time, e2_time, c_time], axis=1)
        observed_time = np.min(all_times, axis=1)
        event_indicator = np.argmin(all_times, axis=1)

        data = np.concatenate([X, observed_time[:, None], event_indicator[:, None],
                               e1_time[:, None], e2_time[:, None], c_time[:, None]], axis=1)
        label_cols = ['observed_time', 'event_indicator', 'e1_time', 'e2_time', 'c_time']
        columns = [f'X{i}' for i in range(n_features)] + label_cols

        df = pd.DataFrame(data, columns=columns)
        self.X = df.drop(label_cols, axis=1)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        self.y_t = np.stack((df['e1_time'].values, df['e2_time'].values, df['c_time'].values), axis=1)
        
        # Encode y_e to CR structure
        events = df['event_indicator'].values.astype('int32')
        num_events = int(np.max(events) + 1)
        self.y_e = np.eye(num_events)[events]
        
        self.params = [beta, beta_shape_e1, beta_scale_e1, beta_shape_e2, beta_scale_e2]
        return self
    
    def split_data(self,
                   train_size: float,
                   valid_size: float,
                   random_state=0):
        # Split multi event data
        raw_data = self.X
        event_time = self.y_t
        labs = self.y_e
        
        traj_labs = labs
        if labs.shape[1] > 1: 
            traj_labs = get_trajectory_labels(labs)

        #split into training/test
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size,
                                          random_state=random_state)
        train_i, test_i = next(splitter.split(raw_data, traj_labs))

        train_data = raw_data.iloc[train_i, :]
        train_labs = labs[train_i, :]
        train_event_time = event_time[train_i, :]

        pretest_data = raw_data.iloc[test_i, :]
        pretest_labs = labs[test_i, :]
        pretest_event_time = event_time[test_i, :]

        #further split test set into test/validation
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=valid_size,
                                          random_state=random_state)
        new_pretest_labs = get_trajectory_labels(pretest_labs)
        test_i, val_i = next(splitter.split(pretest_data, new_pretest_labs))
        test_data = pretest_data.iloc[test_i, :]
        test_labs = pretest_labs[test_i, :]
        test_event_time = pretest_event_time[test_i, :]

        val_data = pretest_data.iloc[val_i, :]
        val_labs = pretest_labs[val_i, :]
        val_event_time = pretest_event_time[val_i, :]

        #package for convenience
        train_pkg = [train_data, train_event_time, train_labs]
        valid_pkg = [val_data, val_event_time, val_labs]
        test_pkg = [test_data, test_event_time, test_labs]
        
        return (train_pkg, valid_pkg, test_pkg)
    
class MultiEventSyntheticDataLoader(BaseDataLoader):
    def load_data(self, copula_parameters=None, dist='Weibull', n_samples=30000,
                  n_features=10, adm_cens_time=5, rng=np.random.default_rng(0)):
        # Generate synthetic data (3 multi events and adm. censoring)
        if copula_parameters is None:
            corrMatrix = np.array([[1, 0.8, 0], [0.8, 1, 0], [0, 0, 1]])
            copula_parameters = [
                {"type": "clayton", "weight": 1 / 4, "theta": 2},
                {"type": "frank", "weight": 1 / 3, "theta": 1},
                {"type": "gumbel", "weight": 1 / 4, "theta": 3}
            ]
        
        X = rng.uniform(0, 1, (n_samples, n_features))
        hidden_dim = int(n_features/2)
        beta = rng.uniform(0, 1, (n_features, hidden_dim))

        beta_shape_e1 = rng.uniform(0, 1, (int(hidden_dim*0.8),))
        beta_scale_e1 = rng.uniform(0, 1, (int(hidden_dim*0.8),))
        beta_shape_e2 = rng.uniform(0, 1, (int(hidden_dim*0.8),))
        beta_scale_e2 = rng.uniform(0, 1, (int(hidden_dim*0.8),))
        beta_shape_e3 = rng.uniform(0, 1, (int(hidden_dim*0.8),))
        beta_scale_e3 = rng.uniform(0, 1, (int(hidden_dim*0.8),))
        
        hidden_rep = np.matmul(X, beta).squeeze()
        hidden_rep = relu(hidden_rep)
        
        shape_e1 = np.matmul(hidden_rep[:, 0:int(hidden_dim*0.8)], beta_shape_e1).squeeze()
        scale_e1 = np.matmul(hidden_rep[:, -int(hidden_dim*0.8):], beta_scale_e1).squeeze()
        shape_e2 = np.matmul(hidden_rep[:, 0:int(hidden_dim*0.8)], beta_shape_e2).squeeze()
        scale_e2 = np.matmul(hidden_rep[:, -int(hidden_dim*0.8):], beta_scale_e2).squeeze()
        shape_e3 = np.matmul(hidden_rep[:, 0:int(hidden_dim*0.8)], beta_shape_e3).squeeze()
        scale_e3 = np.matmul(hidden_rep[:, -int(hidden_dim*0.8):], beta_scale_e3).squeeze()

        u_e1, u_e2, u_e3 = simulation.simu_mixture(3, n_samples, copula_parameters)

        if dist == "Weibull":
            e1_time = inverse_transform_weibull(u_e1, shape_e1, scale_e1)
            e2_time = inverse_transform_weibull(u_e2, shape_e2, scale_e2)
            e3_time = inverse_transform_weibull(u_e3, shape_e3, scale_e3)
        elif dist == "Exp":
            e1_time = inverse_transform_exp(u_e1, shape_e1, scale_e1)
            e2_time = inverse_transform_exp(u_e2, shape_e2, scale_e2)
            e3_time = inverse_transform_exp(u_e3, shape_e3, scale_e3)
        elif dist == "Lognormal":
            e1_time = inverse_transform_lognormal(u_e1, shape_e1, scale_e1)
            e2_time = inverse_transform_lognormal(u_e2, shape_e2, scale_e2)
            e3_time = inverse_transform_lognormal(u_e3, shape_e3, scale_e3)
        else:
            raise ValueError('Dist not implemented')
        
        # Make adm. censoring
        event_times = np.stack([e1_time, e2_time, e3_time], axis=1)
        event_times = np.minimum(event_times, adm_cens_time)
        event_ind = (event_times < adm_cens_time).astype(int)

        # Format data
        columns = [f'X{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=columns)
        self.X = df
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        self.y_t = np.stack((event_times[:,0], event_times[:,1], event_times[:,2]), axis=1)
        self.y_e = np.stack((event_ind[:,0], event_ind[:,1], event_ind[:,2]), axis=1)
        self.n_events = 3
        
        self.params = [beta, beta_shape_e1, beta_scale_e1, beta_shape_e2,
                       beta_scale_e2, beta_shape_e3, beta_scale_e3]
        return self
    
    def split_data(self, train_size: float,
                   valid_size: float,
                   random_state=0):
        # Split multi event data
        raw_data = self.X
        event_time = self.y_t
        labs = self.y_e
        
        traj_labs = labs
        if labs.shape[1] > 1:
            traj_labs = get_trajectory_labels(labs)

        #split into training/test
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size,
                                          random_state=random_state)
        train_i, test_i = next(splitter.split(raw_data, traj_labs))

        train_data = raw_data.iloc[train_i, :]
        train_labs = labs[train_i, :]
        train_event_time = event_time[train_i, :]

        pretest_data = raw_data.iloc[test_i, :]
        pretest_labs = labs[test_i, :]
        pretest_event_time = event_time[test_i, :]

        #further split test set into test/validation
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=valid_size,
                                          random_state=random_state)
        new_pretest_labs = get_trajectory_labels(pretest_labs)
        test_i, val_i = next(splitter.split(pretest_data, new_pretest_labs))
        test_data = pretest_data.iloc[test_i, :]
        test_labs = pretest_labs[test_i, :]
        test_event_time = pretest_event_time[test_i, :]

        val_data = pretest_data.iloc[val_i, :]
        val_labs = pretest_labs[val_i, :]
        val_event_time = pretest_event_time[val_i, :]

        #package for convenience
        train_pkg = [train_data, train_event_time, train_labs]
        valid_pkg = [val_data, val_event_time, val_labs]
        test_pkg = [test_data, test_event_time, test_labs]

        return (train_pkg, valid_pkg, test_pkg)

class ALSDataLoader(BaseDataLoader):
    """
    Data loader for ALS dataset
    """
    def load_data(self, n_samples:int = None):
        df = pd.read_csv(f'{cfg.DATA_DIR}/als.csv', index_col=0)
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
        columns_to_drop = [col for col in df.columns if
                           any(substring in col for substring in ['Observed', 'Event'])]
        df = df.loc[(df['Speech_Observed'] > 0) & (df['Swallowing_Observed'] > 0)
                    & (df['Handwriting_Observed'] > 0) & (df['Walking_Observed'] > 0)] # min time
        df = df.loc[(df['Speech_Observed'] <= 3000) & (df['Swallowing_Observed'] <= 3000)
                    & (df['Handwriting_Observed'] <= 3000) & (df['Walking_Observed'] <= 3000)] # max time
        df = df.dropna(subset=['Handgrip_Strength']) #exclude people with no strength test
        events = ['Speech', 'Swallowing', 'Handwriting', 'Walking']
        self.X = df.drop(columns_to_drop, axis=1)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        times = [df[f'{event_col}_Observed'].values for event_col in events]
        events = [df[f'{event_col}_Event'].values for event_col in events]
        self.y_t = np.stack((times[0], times[1], times[2], times[3]), axis=1)
        self.y_e = np.stack((events[0], events[1], events[2], events[3]), axis=1)
        self.n_events = 4
        return self

    def split_data(self, train_size: float,
                   valid_size: float,
                   random_state=0):
        # Split multi event data
        raw_data = self.X
        event_time = self.y_t
        labs = self.y_e
        
        traj_labs = labs
        if labs.shape[1] > 1:
            traj_labs = get_trajectory_labels(labs)

        #split into training/test
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size,
                                          random_state=random_state)
        train_i, test_i = next(splitter.split(raw_data, traj_labs))

        train_data = raw_data.iloc[train_i, :]
        train_labs = labs[train_i, :]
        train_event_time = event_time[train_i, :]

        pretest_data = raw_data.iloc[test_i, :]
        pretest_labs = labs[test_i, :]
        pretest_event_time = event_time[test_i, :]

        #further split test set into test/validation
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=valid_size,
                                          random_state=random_state)
        new_pretest_labs = get_trajectory_labels(pretest_labs)
        test_i, val_i = next(splitter.split(pretest_data, new_pretest_labs))
        test_data = pretest_data.iloc[test_i, :]
        test_labs = pretest_labs[test_i, :]
        test_event_time = pretest_event_time[test_i, :]

        val_data = pretest_data.iloc[val_i, :]
        val_labs = pretest_labs[val_i, :]
        val_event_time = pretest_event_time[val_i, :]

        #package for convenience
        train_pkg = [train_data, train_event_time, train_labs]
        valid_pkg = [val_data, val_event_time, val_labs]
        test_pkg = [test_data, test_event_time, test_labs]

        return (train_pkg, valid_pkg, test_pkg)

class MimicDataLoader(BaseDataLoader):
    """
    Data loader for MIMIC dataset
    """
    def load_data(self, n_samples:int = None):
        '''
        t and e order, followed by arf, shock, death
        '''
        with open(str(Path(cfg.DATA_DIR)) + "/" + "mimic_dict.pkl", 'rb') as f:
            mimic_dict = pickle.load(f)
        column_names = [f'x_{i}' for i in range(mimic_dict['X'].shape[1])]
        df = pd.DataFrame(mimic_dict['X'], columns=column_names)
        df['T'] = mimic_dict['T']
        df['E'] = mimic_dict['E']
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
        self.X = df[column_names]
        self.y_t = df['T']
        self.y_e = df['E']
        self.n_events = 2
        return self

    def split_data(self,
                   train_size: float,
                   valid_size: float,
                   random_state=0):
        '''
        Since MIMIC one patient has multiple events, we need to split by patients.
        '''
        with open(str(Path(cfg.DATA_DIR)) + "/" + '/mimic_dict.pkl', 'rb') as f:
            mimic_dict = pickle.load(f)
        raw_data = mimic_dict['X']
        event_time = mimic_dict['T']
        labs = mimic_dict['E']        
        not_early = mimic_dict['not_early']
        
        pat_map = pd.read_csv(str(Path(cfg.DATA_DIR)) + "/" + 'pat_to_visit.csv').to_numpy() #
        pat_map = pat_map[not_early, :]
        print('num unique pats', np.unique(pat_map[:, 0]).shape)
        traj_labs = get_trajectory_labels(labs)
        
        #split into training/test
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=train_size,
                                          random_state=random_state)
        train_i, test_i = next(splitter.split(raw_data, traj_labs))
        
        train_pats = pat_map[train_i, 0]
        train_i = np.where(np.isin(pat_map[:, 0], train_pats))[0]
        test_i = np.setdiff1d(np.arange(raw_data.shape[0]), train_i)
    
        train_data = raw_data[train_i, :]
        train_labs = labs[train_i, :]
        train_event_time = event_time[train_i, :]
        
        pretest_data = raw_data[test_i, :]
        pretest_labs = labs[test_i, :]
        pretest_event_time = event_time[test_i, :]
        pretest_pats = pat_map[test_i, 0]
        
        #further split test set into test/validation
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=valid_size,
                                          random_state=random_state)
        test_i, val_i = next(splitter.split(pretest_data, pretest_labs))
        test_pats = pretest_pats[test_i]
        test_i = np.where(np.isin(pretest_pats, test_pats))[0]
        val_i = np.setdiff1d(np.arange(pretest_pats.shape[0]), test_i)
    
        test_data = pretest_data[test_i, :]
        test_labs = pretest_labs[test_i, :]
        test_event_time = pretest_event_time[test_i, :]
        
        val_data = pretest_data[val_i, :]
        val_labs = pretest_labs[val_i, :]
        val_event_time = pretest_event_time[val_i, :]
        
        #package for convenience
        train_package = [train_data, train_event_time, train_labs]
        test_package = [test_data, test_event_time, test_labs]
        validation_package = [val_data, val_event_time, val_labs]
    
        return train_package, test_package, validation_package

class SeerDataLoader(BaseDataLoader):
    """
    Data loader for SEER dataset
    """
    def load_data(self, n_samples:int = None):
        df = pd.read_csv(f'{cfg.DATA_DIR}/seer_processed.csv')
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
        self.X = df.drop(['duration', 'event_heart', 'event_breast'], axis=1)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        events = ['heart', 'breast']
        events = [df[f'event_{event_col}'].values for event_col in events]
        self.y_t = np.stack((df[f'duration'].values, df[f'duration'].values), axis=1)
        self.y_e = np.stack((events[0], events[1]), axis=1)
        self.n_events = 2
        return self
    
    def split_data(self,
                   train_size: float,
                   valid_size: float,
                   random_state=0):
        # Split multi event data
        raw_data = self.X
        event_time = self.y_t
        labs = self.y_e
        
        traj_labs = labs
        if labs.shape[1] > 1: 
            traj_labs = get_trajectory_labels(labs)

        #split into training/test
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size,
                                          random_state=random_state)
        train_i, test_i = next(splitter.split(raw_data, traj_labs))

        train_data = raw_data.iloc[train_i, :]
        train_labs = labs[train_i, :]
        train_event_time = event_time[train_i, :]

        pretest_data = raw_data.iloc[test_i, :]
        pretest_labs = labs[test_i, :]
        pretest_event_time = event_time[test_i, :]

        #further split test set into test/validation
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=valid_size,
                                          random_state=random_state)
        new_pretest_labs = get_trajectory_labels(pretest_labs)
        test_i, val_i = next(splitter.split(pretest_data, new_pretest_labs))
        test_data = pretest_data.iloc[test_i, :]
        test_labs = pretest_labs[test_i, :]
        test_event_time = pretest_event_time[test_i, :]

        val_data = pretest_data.iloc[val_i, :]
        val_labs = pretest_labs[val_i, :]
        val_event_time = pretest_event_time[val_i, :]

        #package for convenience
        train_pkg = [train_data, train_event_time, train_labs]
        valid_pkg = [val_data, val_event_time, val_labs]
        test_pkg = [test_data, test_event_time, test_labs]

        return (train_pkg, valid_pkg, test_pkg)

class RotterdamDataLoader(BaseDataLoader):
    """
    Data loader for Rotterdam dataset
    """
    def load_data(self, n_samples:int = None):
        df = pd.read_csv(f'{cfg.DATA_DIR}/rotterdam.csv')
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
        self.X = df.drop(['pid', 'rtime', 'recur', 'dtime', 'death'], axis=1)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        times = [df['rtime'].values, df['dtime'].values]
        events = [df['recur'].values, df['death'].values]
        self.y_t = np.stack((times[0], times[1]), axis=1)
        self.y_e = np.stack((events[0], events[1]), axis=1)
        self.n_events = 2
        return self
    
    def split_data(self,
                   train_size: float,
                   valid_size: float,
                   random_state=0):
        # Split multi event data
        raw_data = self.X
        event_time = self.y_t
        labs = self.y_e
        
        traj_labs = labs
        if labs.shape[1] > 1: 
            traj_labs = get_trajectory_labels(labs)

        #split into training/test
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size,
                                          random_state=random_state)
        train_i, test_i = next(splitter.split(raw_data, traj_labs))

        train_data = raw_data.iloc[train_i, :]
        train_labs = labs[train_i, :]
        train_event_time = event_time[train_i, :]

        pretest_data = raw_data.iloc[test_i, :]
        pretest_labs = labs[test_i, :]
        pretest_event_time = event_time[test_i, :]

        #further split test set into test/validation
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=valid_size)
        new_pretest_labs = get_trajectory_labels(pretest_labs)
        test_i, val_i = next(splitter.split(pretest_data, new_pretest_labs))
        test_data = pretest_data.iloc[test_i, :]
        test_labs = pretest_labs[test_i, :]
        test_event_time = pretest_event_time[test_i, :]

        val_data = pretest_data.iloc[val_i, :]
        val_labs = pretest_labs[val_i, :]
        val_event_time = pretest_event_time[val_i, :]

        #package for convenience
        train_pkg = [train_data, train_event_time, train_labs]
        valid_pkg = [val_data, val_event_time, val_labs]
        test_pkg = [test_data, test_event_time, test_labs]

        return (train_pkg, valid_pkg, test_pkg)
    
def get_data_loader(dataset_name:str) -> BaseDataLoader:
    if dataset_name == "seer":
        return SeerDataLoader()
    elif dataset_name == "als":
        return ALSDataLoader()
    elif dataset_name == "mimic":
        return MimicDataLoader()
    elif dataset_name == "rotterdam":
        return RotterdamDataLoader()
    else:
        raise ValueError("Dataset not found")
        