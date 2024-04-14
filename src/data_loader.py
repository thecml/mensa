import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from typing import Tuple, List
from preprocessor import Preprocessor
from pathlib import Path
import config as cfg
from utility.survival import convert_to_structured
from utility.data import make_synthetic
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from utility.survival import get_trajectory_labels

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

    @abstractmethod
    def load_data(self) -> None:
        """Loads the data from a data set at startup"""

    def get_data(self) -> pd.DataFrame:
        """
        This method returns the features and targets
        :return: df
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
        return data.select_dtypes(['category']).columns.tolist()

    def split_data(self, test_size: float=0.4, valid_size: float=0.5) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training, validation, and test sets (from Donna's paper)
        """
        raw_data = self.X
        event_time = self.y_t
        labs = self.y_e
        
        traj_labs = labs
        if labs.shape[1] > 1: 
            traj_labs = get_trajectory_labels(labs)

        #split into training/test
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
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
        train_package = [train_data, train_event_time, train_labs]
        test_package = [test_data, test_event_time, test_labs]
        validation_package = [val_data, val_event_time, val_labs]

        return train_package, test_package, validation_package

class SyntheticDataLoader(BaseDataLoader):
    """
    Data loader for synthetic data
    """
    def load_data(self):
        params = cfg.SYNTHETIC_SETTINGS
        raw_data, event_times, labs = make_synthetic(params['num_events'])
        if params['discrete'] == False:
            min_time = np.min(event_times[event_times != -1]) 
            max_time = np.max(event_times[event_times != -1]) 
            time_range = max_time - min_time
            bin_size = time_range / params['num_bins']
            binned_event_time = np.floor((event_times - min_time) / bin_size)
            binned_event_time[binned_event_time == params['num_bins']] = params['num_bins'] - 1 
        self.X = pd.DataFrame(raw_data)
        self.y_t = binned_event_time
        self.y_e = labs
        self.min_time = min_time
        self.max_time = max_time
        return self

class ALSDataLoader(BaseDataLoader):
    """
    Data loader for ALS dataset
    """
    def load_data(self):
        df = pd.read_csv(f'{cfg.DATA_DIR}/als.csv')
        columns_to_drop = [col for col in df.columns if
                           any(substring in col for substring in ['Observed', 'Event'])]
        events = ['Speech', 'Swallowing', 'Handwriting', 'Walking']
        self.X = df.drop(columns=columns_to_drop)
        times = [df[f'{event_col}_Observed'].values for event_col in events]
        events = [df[f'{event_col}_Event'].values for event_col in events]
        self.y_t = np.stack((times[0], times[1], times[2], times[3]), axis=1)
        self.y_e = np.stack((events[0], events[1], events[2], events[3]), axis=1)
        return self

class MimicDataLoader(BaseDataLoader):
    """
    Data loader for MIMIC dataset
    """
    def load_data(self):
        pass

class SeerDataLoader(BaseDataLoader):
    """
    Data loader for SEER dataset
    """
    def load_data(self):
        pass
    
