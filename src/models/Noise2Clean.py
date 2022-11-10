import torch
import torch.nn as nn


# median absolute deviation
# used as an analog of the noise level map
class MedianAbsoluteDeviation(nn.Module):
    def __init__(self, ksize=3, down=False):
        super(MedianAbsoluteDeviation, self).__init__()
        self.ksize = ksize
        self.down = down
        self.reflect = nn.ReflectionPad2d(ksize // 2)
        self.Avg = None
        self.Ups = None
        if down:
            self.Avg = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
            self.Ups = nn.Upsample(scale_factor=2, mode='bilinear')

    def median(self, img):
        img = self.reflect(img).unfold(2, self.ksize, 1).unfold(3, self.ksize, 1)
        img = img.contiguous().view(img.size()[:4] + (-1,)).median(dim=-1)[0]
        return img

    def compute_mad(self, img):
        # find median of image
        median_img = self.median(img)  # this function includes padding inside it
        img = self.reflect(img).unfold(2, self.ksize, 1).unfold(3, self.ksize, 1)

        # transform images into ksize x ksize tiles
        median_img_t = torch.repeat_interleave(median_img, self.ksize * self.ksize, dim=3).flatten().reshape(img.size())

        z = torch.abs(img - median_img_t)
        # and now take the median within each tile
        mad = z.contiguous().view(z.size()[:4] + (-1,)).median(dim=-1)[0]
        return mad

    def forward(self, x):
        center_idx = x.shape[1] // 2
        if self.down:
            center = self.Avg(x[:, center_idx:(center_idx + 1), :, :])
        else:
            center = x[:, center_idx:(center_idx + 1), :, :]
        bs = center.shape[0]
        mad = torch.zeros_like(center)
        for i in range(0, bs):
            mad[i:(i + 1), :, :, :] = self.compute_mad(center[i:(i + 1), :, :, :])
        if self.down:
            return self.Ups(mad)
        return mad


class DnCNN(nn.Module):
    def __init__(self, channels, out_channel=None, num_of_layers=8):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        padding_mode = 'replicate'
        features = 64
        layers = []
        if not out_channel:
            out_channel = channels
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features,
                                kernel_size=kernel_size, bias=False,
                                padding=padding, padding_mode=padding_mode))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features,
                                    kernel_size=kernel_size, bias=False,
                                    padding=padding, padding_mode=padding_mode))
            # layers.append(nn.BatchNorm2d(features))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=out_channel,
                                kernel_size=kernel_size, bias=False,
                                padding=padding, padding_mode=padding_mode))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out
