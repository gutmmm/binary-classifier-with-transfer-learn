import torch.nn as nn
from torch.nn.functional import max_pool2d


class BinaryFMnist(nn.Module):
    def __init__(self):
        super(BinaryFMnist, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.relu = nn.LeakyReLU()
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=1)

    def forward(self, x):
        # conv 1
        x = self.conv1(x)
        x = self.relu(x)
        x = max_pool2d(x, kernel_size=2, stride=2)
        # conv 2
        x = self.conv2(x)
        x = self.relu(x)
        x = max_pool2d(x, kernel_size=2, stride=2)
        # fc1
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        # fc2
        x = self.linear2(x)
        x = self.relu(x)
        # output
        x = self.out(x)

        return x