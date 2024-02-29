import torch.nn as nn
import torch.optim as optim


class BenchmarkCnn2(nn.Module):
    def __init__(self, image_size, num_classes):
        super(BenchmarkCnn2, self).__init__()
        self.name = "benchmark_cnn"
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Conv2d(32, 32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(28 * 28, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        lr_schedule = optim.lr_scheduler.ExponentialLR(
            optim.Adam(self.parameters(), lr=0.0001), gamma=0.9, verbose=True
        )
        return optim.Adam(self.parameters(), lr=0.0001), lr_schedule
