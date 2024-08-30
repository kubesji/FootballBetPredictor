import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np


FC1, FC2, FC3, FC4 = 16, 16, 16, 8


class FBM(nn.Module):
    def __init__(self, in_size, num_class, dropout=0):
        super(FBM, self).__init__()
        self.fc1 = nn.Linear(in_size, FC1)
        self.fc2 = nn.Linear(FC1, FC2)
        self.fc3 = nn.Linear(FC2, FC3)
        self.fc4 = nn.Linear(FC3, FC4)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.out = nn.Linear(FC4, num_class)
        self.init_weights()

        self.use_dropout = True if dropout > 0 else False

    def init_weights(self):
        INITRANGE = 0.1
        self.fc1.weight.data.uniform_(-INITRANGE, INITRANGE)
        self.fc2.weight.data.uniform_(-INITRANGE, INITRANGE)
        self.fc3.weight.data.uniform_(-INITRANGE, INITRANGE)
        self.fc4.weight.data.uniform_(-INITRANGE, INITRANGE)
        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()
        self.fc3.bias.data.zero_()
        self.fc4.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        if self.use_dropout:
            x = self.dropout4(x)
        x = F.softmax(self.out(x), dim=1)
        return x


class DatasetFBM(Dataset):
    def __init__(self, X, Y, device):
        self.X = X
        self.Y = Y
        self.len = len(Y)
        self.device = device

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx]

    def __len__(self):
        return self.len

    def collate_batch(self, batch):
        X, Y = [], []
        for x, y in batch:
            tmp = [1 if i == y else 0 for i in range(3)]
            Y.append(tmp)
            X.append(x)
        Y = torch.tensor(np.array(Y), dtype=torch.float)
        X = torch.tensor(np.array(X), dtype=torch.float)
        return X.to(self.device), Y.to(self.device)
