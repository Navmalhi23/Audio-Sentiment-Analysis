import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEmotionClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (1, 43, 400) -> (16, 43, 400)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                            # (16, 21, 200)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (32, 21, 200)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                            # (32, 10, 100)
        )

        # flatten = 32 * 10 * 100 = 32000
        self.fc = nn.Sequential(
            nn.Linear(32000, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
