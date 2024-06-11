import torch.nn as nn
import torch


class ResBlock(nn.Module):
    def __init__(self, ni, nf):
        super(ResBlock, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv1d(ni, nf, 7, padding='same', padding_mode='reflect'),
            nn.BatchNorm1d(nf),
            nn.ReLU()
        )
        self.convblock2 = nn.Sequential(
            nn.Conv1d(nf, nf, 5, padding='same', padding_mode='reflect'),
            nn.BatchNorm1d(nf),
            nn.ReLU()
        )
        self.convblock3 = nn.Sequential(
            nn.Conv1d(nf, nf, 3, padding='same', padding_mode='reflect'),
            nn.BatchNorm1d(nf),
            nn.ReLU()
        )
        self.shortcut = nn.BatchNorm1d(ni) if ni == nf else nn.Conv1d(ni, nf, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = torch.add(x, self.shortcut(res))
        x = self.act(x)
        return x


class ResNet(nn.Module):
    def __init__(self, c_in, n_classes):
        super(ResNet, self).__init__()
        # Block1
        nf = 64
        self.res_block1 = ResBlock(c_in, nf)
        self.res_block2 = ResBlock(nf, nf * 2)
        self.res_block3 = ResBlock(nf * 2, nf * 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nf * 2, n_classes)

    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.gap(x)
        x = x.squeeze()
        x = self.fc(x)
        # x = x.view(x.size(0), -1)
        return x

x = torch.randn(16, 1, 100)
model = ResNet(1, 2)
x = model(x)
print(x.shape)