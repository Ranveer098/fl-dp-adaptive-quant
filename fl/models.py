"""Model architectures from paper Section III-C (Figures 2 and 3)."""
import torch.nn as nn


class MNISTCNN(nn.Module):
    """2-layer CNN from the FedAvg paper: conv(32,5x5)-pool-conv(64,5x5)-pool-FC512-FC10.
    1,663,370 parameters (paper Table 2: 1.6M)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1, 2), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 512), nn.ReLU(inplace=True), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _block(cin, cout):
    return [nn.Conv2d(cin, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class VGG7(nn.Module):
    """VGG7 (vgg7_bn in the authors' code, fedml/model/cv/cnn.py):
    32x32x3 -> 16x16x32 -> 8x8x64 -> 8x8x128 -> 4x4x128 -> avgpool 2x2x128 -> 512 -> 128 -> 128 -> 10.
    325,002 parameters + 704 BatchNorm running statistics (paper Table 2: 300K)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            *_block(3, 32), nn.MaxPool2d(2),
            *_block(32, 64), nn.MaxPool2d(2),
            *_block(64, 128),
            *_block(128, 128), nn.MaxPool2d(2),  # this extra pool is in the official code
            nn.AdaptiveAvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(dataset: str):
    return {"mnist": MNISTCNN, "cifar10": VGG7}[dataset]()
