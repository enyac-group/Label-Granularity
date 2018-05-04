'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),  # b, 16, 32, 32
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 16, 16
            nn.Conv2d(16, 8, 3, stride=1, padding=1),  # b, 8, 16, 16
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 8, 8, 8
            nn.Conv2d(8, 4, 3, stride=1, padding=1),  # b, 4, 8, 8
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 4, 4, 4
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 3, stride=2, padding=1),  # b, 16, 7, 7
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=0),  # b, 16, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 4, stride=2, padding=0),  # b, 8, 32, 32
            nn.Tanh(),
        )

    def forward(self, x):
        feats = self.encoder(x)
        x = self.decoder(feats)
        feats = feats.view(feats.size(0), -1)
        return x, feats


def Auto_encoder():
    return autoencoder()


