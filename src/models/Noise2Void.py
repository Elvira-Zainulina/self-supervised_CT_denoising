import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate(im, angle):
    if angle == 0:
        return im
    elif angle == 90:
        return im.transpose(-1, -2).flip(-1)
    elif angle == 180:
        return im.flip(-1).flip(-2)
    elif angle == 270:
        return im.transpose(-1, -2).flip(-2)


class Conv(nn.Module):
    """
        Special convolution proposed in "High quality self-supervised deep image denoising"
        (http://arxiv.org/abs/1901.10277)
    """
    def __init__(self, in_dim, out_dim, kernel_size, bias=False,
                 padding_mode='zeros'):
        super(Conv, self).__init__()

        assert padding_mode in [None, 'zeros', 'reflect', 'replicate', 'circular']

        self.offset = kernel_size // 2
        self.padding = kernel_size // 2
        self.padding_mode = padding_mode

        if padding_mode is None:
            self.padding, self.padding_mode = 0, 'constant'
        if padding_mode == 'zeros':
            self.padding_mode = 'constant'

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, bias=bias)
        nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        x = F.pad(x, [self.padding] * 2 + [self.padding + self.offset] + [self.padding],
                  mode=self.padding_mode)
        x = self.conv(x)
        if self.offset > 0:
            return x[:, :, :-self.offset, :]
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, num_convs=1,
                 bias=False, padding_mode='zeros'):
        super().__init__()

        assert padding_mode in [None, 'zeros', 'reflect', 'replicate', 'circular']

        self.convs = []
        self.convs.append(Conv(in_dim, out_dim, kernel_size=kernel_size,
                               padding_mode=padding_mode))
        self.convs.append(nn.LeakyReLU(0.1))
        for _ in range(num_convs - 1):
            self.convs.append(Conv(out_dim, out_dim,
                                   kernel_size=kernel_size,
                                   padding_mode=padding_mode, bias=bias))
            self.convs.append(nn.LeakyReLU(0.1))
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        return self.convs(x)


# Model based on DnCNN neural network architecture
class Noise2VoidDnCNN(nn.Module):
    def __init__(self, in_dim=1, out_dim=2, kernel_size=3,
                 n_channels=64, n_layers=5, padding_mode='replicate'):
        super(Noise2VoidDnCNN, self).__init__()

        self.dncnn = []
        self.dncnn.append(Conv(in_dim, n_channels, kernel_size,
                               padding_mode=padding_mode))
        self.dncnn.append(nn.LeakyReLU(0.1))
        for _ in range(n_layers - 1):
            self.dncnn.append(Conv(n_channels, n_channels, kernel_size,
                                   padding_mode=padding_mode))
            self.dncnn.append(nn.LeakyReLU(0.1))
        self.dncnn = nn.Sequential(*self.dncnn)

        self.final_block = nn.Sequential(
            ConvBlock(n_channels * 4, n_channels, num_convs=2, kernel_size=1),
            Conv(n_channels, out_dim, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        outputs = []
        for a in [0, 90, 180, 270]:
            out = rotate(x, a)
            out = self.dncnn(out)

            out = F.pad(out[:, :, :-1, :], [0, 0, 1, 0])
            if a != 0:
                out = rotate(out, 360 - a)
            outputs += [out]

        out = torch.cat(outputs, dim=1)
        out = self.final_block(out)
        return out
