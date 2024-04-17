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

    @abstractmethod
    def load_data(self) -> None:
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

class ALSDataLoader(BaseDataLoader):
    """
    Data loader for ALS dataset
    """
    def load_data(self):
        df = pd.read_csv(f'{cfg.DATA_DIR}/als.csv', index_col=0)
        columns_to_drop = [col for col in df.columns if
                           any(substring in col for substring in ['Observed', 'Event'])]
        df = df.loc[(df['Speech_Observed'] > 0) & (df['Swallowing_Observed'] > 0)
                    & (df['Handwriting_Observed'] > 0) & (df['Walking_Observed'] > 0)] # min time
        df = df.loc[(df['Speech_Observed'] <= 3000) & (df['Swallowing_Observed'] <= 3000)
                    & (df['Handwriting_Observed'] <= 3000) & (df['Walking_Observed'] <= 3000)] # max time
        #df = df.dropna(subset=['Handgrip_Strength']) #exclude people with no strength test
        events = ['Speech', 'Swallowing', 'Handwriting', 'Walking']
        self.X = df.drop(columns_to_drop, axis=1)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        times = [df[f'{event_col}_Observed'].values for event_col in events]
        events = [df[f'{event_col}_Event'].values for event_col in events]
        self.y_t = np.stack((times[0], times[1], times[2], times[3]), axis=1)
        self.y_e = np.stack((events[0], events[1], events[2], events[3]), axis=1)
        self.n_events = self.y_e.shape[1]
        return self

    def split_data(self, train_size: float, valid_size: float):
        # Split multi event data
        raw_data = self.X
        event_time = self.y_t
        labs = self.y_e
        
        traj_labs = labs
        if labs.shape[1] > 1: 
            traj_labs = get_trajectory_labels(labs)

        #split into training/test
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size)
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

class MimicDataLoader(BaseDataLoader):
    """
    Data loader for MIMIC dataset
    """
    def load_data(self):
        '''
        t and e order, followed by arf, shock, death
        '''
        with open(str(Path(cfg.DATA_DIR)) + "/" + "mimic_dict.pkl", 'rb') as f:
            mimic_dict = pickle.load(f)
        column_names = [f'x_{i}' for i in range(mimic_dict['X'].shape[1])]

        self.X = pd.DataFrame(mimic_dict['X'], columns=column_names)
        self.y_t = mimic_dict['T']
        self.y_e = mimic_dict['E']        
        return self

    def split_data(self,
                   train_size: float,
                   valid_size: float):
        '''
        Since MIMIC one patient has multiple events, we need to split by patients.
        '''
        with open(cfg.DATA_DIR+'/mimic_dict.pkl', 'rb') as f:
            mimic_dict = pickle.load(f)
        raw_data = mimic_dict['X']
        event_time = mimic_dict['T']
        labs = mimic_dict['E']        
        not_early = mimic_dict['not_early']
        
        pat_map = pd.read_csv(cfg.DATA_DIR+'/pat_to_visit.csv').to_numpy() #
        pat_map = pat_map[not_early, :]
        print('num unique pats', np.unique(pat_map[:, 0]).shape)
        traj_labs = get_trajectory_labels(labs)
        
        #split into training/test
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=train_size)
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
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=valid_size)
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
    def load_data(self):
        df = pd.read_csv(f'{cfg.DATA_DIR}/seer_processed.csv')
        self.X = df.drop(['duration', 'event_heart', 'event_breast'], axis=1)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        events = ['heart', 'breast']
        events = [df[f'event_{event_col}'].values for event_col in events]
        self.y_t = np.stack((df[f'duration'].values, df[f'duration'].values), axis=1)
        self.y_e = np.stack((events[0], events[1]), axis=1)
        return self
    
    def split_data(self,
                   train_size: float,
                   valid_size: float):
        # Split multi event data
        raw_data = self.X
        event_time = self.y_t
        labs = self.y_e
        
        traj_labs = labs
        if labs.shape[1] > 1: 
            traj_labs = get_trajectory_labels(labs)

        #split into training/test
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size)
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

class RotterdamDataLoader(BaseDataLoader):
    """
    Data loader for Rotterdam dataset
    """
    def load_data(self):
        df = pd.read_csv(f'{cfg.DATA_DIR}/rotterdam.csv')
        self.X = df.drop(['rtime', 'recur', 'dtime', 'death'], axis=1)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        times = [df['rtime'].values, df['dtime'].values]
        events = [df['recur'].values, df['death'].values]
        self.y_t = np.stack((times[0], times[1]), axis=1)
        self.y_e = np.stack((events[0], events[1]), axis=1)
        return self
    
    def split_data(self,
                   train_size: float,
                   valid_size: float):
        pass
    
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
        