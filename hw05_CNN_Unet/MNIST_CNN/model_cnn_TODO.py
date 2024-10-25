import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.pool1 = nn.MaxPool2d(2)


        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 第一层卷积
        x = F.relu(self.conv1(x))
        # x = self.pool1(x)
        # 第二层卷积
        x = F.relu(self.conv2(x))
        # 池化层
        x = self.pool1(x)
        # 全连接层 1
        x = torch.flatten(x,1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        # 全连接层 2 （输出层）
        x = self.dropout2(x)
        x = self.fc2(x)
        out = x
        
        assert out.ndim == 2 and out.size(0) == x.size(0) and out.size(1) == 10

        return out
