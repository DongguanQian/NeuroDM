import torch
import torch.nn as nn
from einops import rearrange


# Convolutional Feature Expansion
class CFE1(nn.Module):
    def __init__(self, channels, E=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, E // 2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(channels, E // 2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.bn = nn.BatchNorm2d(E)
        self.elu = nn.ELU()
        self.conv3 = nn.Conv2d(E, channels, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        # [b, c, h, w]
        x1 = self.conv1(x)  # [b, E/2, h, w]
        x2 = self.conv2(x)  # [b, E/2, h, w]
        x = torch.cat((x1, x2), dim=1)  # [b, E, h, w]

        x = self.bn(x)
        x = self.elu(x)
        x = self.conv3(x)  # [b, c, h, w]
        return x


class CFE2(nn.Module):
    def __init__(self, channels, E=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, E // 2, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.conv2 = nn.Conv2d(channels, E // 2, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.bn = nn.BatchNorm2d(E)
        self.elu = nn.ELU()
        self.conv3 = nn.Conv2d(E, channels, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        # [b, c, h, w]
        x1 = self.conv1(x)  # [b, E/2, h, w]
        x2 = self.conv2(x)  # [b, E/2, h, w]
        x = torch.cat((x1, x2), dim=1)  # [b, E, h, w]

        x = self.bn(x)
        x = self.elu(x)
        x = self.conv3(x)  # [b, c, h, w]
        return x


class MHA(nn.Module):
    def __init__(self, channels, num_heads=2):
        super().__init__()
        self.h = num_heads
        self.d = channels // num_heads
        # scale factor
        self.scale = self.d ** -0.5

        self.conv_qkv = nn.Conv2d(in_channels=channels, out_channels=3 * channels, kernel_size=(1, 1), stride=(1, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # [b, c, p, t]
        qkv = self.conv_qkv(x)  # [b, c, p, t] -> [b, 3*c, p, t]
        q, k, v = rearrange(qkv, 'b (qkv h d) p t -> qkv b h d p t', qkv=3, h=self.h, d=self.d)
        q = rearrange(q, 'b h d p t -> b h p (d t)')
        k = rearrange(k, 'b h d p t -> b h (d t) p')
        v = rearrange(v, 'b h d p t -> b h p (d t)')

        dots = torch.matmul(q, k) * self.scale  # [b, h, p, p]
        attn = self.softmax(dots)

        out = torch.matmul(attn, v)  # [b, h, p, (dt)]
        out = rearrange(out, 'b h p (d t) -> b (h d) p t', h=self.h, d=self.d)
        return out


class CTBlock1(nn.Module):
    def __init__(self, channels, num_heads, E):
        super().__init__()
        self.mha = MHA(channels=channels, num_heads=num_heads)
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.cfe = CFE1(channels=channels, E=E)
        self.bn2 = nn.BatchNorm2d(num_features=channels)

    def forward(self, x):
        # [b, c, h, w]
        x = torch.add(self.mha(x), x)
        x = self.bn1(x)
        x = torch.add(self.cfe(x), x)
        x = self.bn2(x)
        return x


class CTBlock2(nn.Module):
    def __init__(self, channels, num_heads, E):
        super().__init__()
        self.mha = MHA(channels=channels, num_heads=num_heads)
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.cfe = CFE2(channels=channels, E=E)
        self.bn2 = nn.BatchNorm2d(num_features=channels)

    def forward(self, x):
        # [b, c, h, w]
        x = torch.add(self.mha(x), x)
        x = self.bn1(x)
        x = torch.add(self.cfe(x), x)
        x = self.bn2(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.ctblock1 = CTBlock1(channels=64, num_heads=4, E=16)
        self.ctblock2 = CTBlock2(channels=256, num_heads=8, E=64)

        self.spatial_temporal = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(512),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(512, 40)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, x.size(1), x.size(2))  # [b, 1, 440, 128]
        x = self.temporal(x)
        x = self.ctblock1(x)
        x = self.spatial(x)
        x = self.ctblock2(x)
        x = self.spatial_temporal(x).view(batch_size, -1)

        y = self.fc(x)

        return y
