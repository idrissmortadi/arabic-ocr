import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

class HandwritingRecognizer1(nn.Module):
    def __init__(self, image_width, image_height, vocabulary_size):
        super(HandwritingRecognizer1, self).__init__()
        self.name = "handwriting_recognizer_1"
        self.image_width = image_width
        self.image_height = image_height

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.dense1 = nn.Linear(image_width // 2 * image_height // 2 * 32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.lstm = nn.LSTM(16, 128, bidirectional=True, dropout=0.35, batch_first=True)
        self.dense2 = nn.Linear(256, vocabulary_size + 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = x.view(-1, self.image_width // 2 * self.image_height // 2 * 32)
        x = F.relu(self.dense1(x))
        x = self.bn2(x)
        x = x.unsqueeze(2)
        x, _ = self.lstm(x)
        x = self.dense2(x)
        # Include log_softmax for compatibility with CTC loss
        return F.log_softmax(x, dim=2)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return optimizer, lr_scheduler
