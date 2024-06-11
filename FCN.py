import torch
from torch import nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self, c_in, c_out):
        super(FCN, self).__init__()
        self.cov_block1 = nn.Sequential(
            nn.Conv1d(c_in, 128, 7, padding='same', padding_mode='reflect'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.cov_block2 = nn.Sequential(
            nn.Conv1d(128, 256, 5, padding='same', padding_mode='reflect'),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.cov_block3 = nn.Sequential(
            nn.Conv1d(256, 128, 3, padding='same', padding_mode='reflect'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, c_out)

    def forward(self, x):
        x = self.cov_block1(x)
        x = self.cov_block2(x)
        x = self.cov_block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
