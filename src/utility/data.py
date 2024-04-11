import math

import pylab
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MultiEventDataset(Dataset):

    def __init__(self, feature_num, X, Y_T, Y_E):

        self.feature_num = feature_num

        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y1 = torch.tensor(np.stack((Y_T[:,0], Y_E[:,0]), axis=1), dtype=torch.float32)
        self.Y2 = torch.tensor(np.stack((Y_T[:,1], Y_E[:,1]), axis=1), dtype=torch.float32)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx,:], self.Y1[idx,:], self.Y2[idx,:]