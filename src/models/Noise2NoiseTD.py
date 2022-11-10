import torch
import torch.nn as nn


# LSTM layers
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding_mode='zeros'):
        super(ConvLSTMCell, self).__init__()

        assert padding_mode in ['zeros', 'reflect', 'replicate', 'circular']

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(input_dim + hidden_dim, hidden_dim * 4,
                              kernel_size, bias=False,
                              padding=kernel_size // 2,
                              padding_mode=padding_mode)
        nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, input_tensor, cur_state):  # inputs of size [BxCxHxW] * 3
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined = self.conv(combined)
        cc_i, cc_f, cc_o, cc_c = torch.split(combined, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)

        c_next = f * c_cur + i * torch.tanh(cc_c)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next  # outputs of size [Bxhidden_dimxHxW] * 2

    def init_hidden(self, batch_size, shape):
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, shape[0], shape[1],
                            requires_grad=True, device=device),
                torch.zeros(batch_size, self.hidden_dim, shape[0], shape[1],
                            requires_grad=True, device=device))


class SEblock(nn.Module):
    """credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"""

    def __init__(self, c, r=8):
        super().__init__()
        self.c = c
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.squeeze(x).view(*x.shape[:-2])
        y = self.excitation(y).view(*x.shape[:-2], 1, 1)
        return x * y.expand_as(x)


class FinalConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, seq_len=6,
                 kernel_size=3, padding_mode='zeros'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode

        self.cells = ConvLSTMCell(input_dim=input_dim,
                                  hidden_dim=self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding_mode=self.padding_mode)
        self.se = SEblock(seq_len, r=2)

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: 5-D Tensor of shape (b, t, c, h, w)

        Returns:
            tensor of shape BxTxCxHxW
        """

        b, seq_len, _, h, w = input_tensor.size()

        h, c = self.init_hidden(batch_size=b, shape=(h, w))

        outf = []
        outb = []

        input_tensor = torch.split(input_tensor, seq_len // 2, 1)
        input_tensor = torch.cat(input_tensor, 0)
        # print(input_tensor.shape)
        for t in range(seq_len // 2):
            h, c = self.cells(input_tensor=input_tensor[:, t, ...],
                              cur_state=[h, c])
            h1 = torch.split(h, b, 0)
            outf.append(h1[0])
            outb.append(h1[1])

        outf = torch.stack(outf, dim=1)  # BxT//2xCxHxW
        outb = torch.stack(outb, dim=1)  # BxT//2xCxHxW
        out = torch.cat([outf, outb], 1)  # BxTxCxHxW
        out = self.se(out.transpose(1, 2)).transpose(1, 2)

        return out  # BxTxCxHxW

    def init_hidden(self, batch_size, shape):
        return self.cells.init_hidden(batch_size * 2, shape)


# Model based on DnCNN neural network architecture
class N2NTDDnCNN(nn.Module):
    def __init__(self, in_dim=1, out_dim=2, kernel_size=3, seq_len=6,
                 n_channels=64, n_layers=5, padding_mode='replicate'):
        super().__init__()

        self.out_dim = out_dim
        self.seq_len = seq_len

        self.dncnn = []
        self.dncnn.append(nn.Conv2d(in_dim, n_channels, kernel_size,
                                    padding=kernel_size // 2, bias=False,
                                    padding_mode=padding_mode))
        self.dncnn.append(nn.LeakyReLU(0.1))

        for _ in range(n_layers - 1):
            self.dncnn.append(nn.Conv2d(n_channels, n_channels, kernel_size,
                                        padding=kernel_size // 2, bias=False,
                                        padding_mode=padding_mode))
            self.dncnn.append(nn.LeakyReLU(0.1))
        self.dncnn.append(nn.Conv2d(n_channels, n_channels // 4, kernel_size,
                                    padding=kernel_size // 2, bias=False,
                                    padding_mode=padding_mode))
        self.dncnn.append(nn.LeakyReLU(0.1))
        self.dncnn = nn.Sequential(*self.dncnn)

        self.lstm = FinalConvLSTM(n_channels // 4, n_channels // 4, kernel_size=kernel_size,
                                  padding_mode=padding_mode, seq_len=seq_len)

        self.final_block = nn.Sequential(
            nn.Conv2d(n_channels // 4 * seq_len, n_channels, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(n_channels, out_dim, kernel_size=1, bias=False),
            nn.ReLU()
        )

        for i in range(n_layers + 1):
            nn.init.kaiming_uniform_(self.dncnn[i * 2].weight)
        nn.init.kaiming_uniform_(self.final_block[0].weight)

    def forward(self, x):

        batch_size, seq_len = x.shape[:2]

        out = x.reshape(batch_size * seq_len, *x.shape[2:])
        out = self.dncnn(out)
        out = out.reshape(batch_size, seq_len, *out.shape[1:])

        n_channels = out.shape[2]
        out = self.lstm(out)
        out = out.reshape(batch_size, seq_len * n_channels, *out.shape[3:])
        out = self.final_block(out)
        return out
