import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyWakeWordCNN(nn.Module):
    def __init__(self):
        super(TinyWakeWordCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.dropout(x, 0.2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.dropout(x, 0.2)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    

class TinyWakeWordLSTM(nn.Module):
    def __init__(self):
        super(TinyWakeWordLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=4,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = x.squeeze(1)
        _, (h, _) = self.lstm(x)
        x = h[-1]
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x