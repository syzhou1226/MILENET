import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class Swish(nn.Module):
    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x * torch.sigmoid(x)
        else:
            return x * torch.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class AttentionModule(nn.Module):
    def __init__(self, mid_d):
        super(AttentionModule, self).__init__()
        self.mid_d = mid_d

        self.coordatt = CoordAtt(inp=self.mid_d, oup=self.mid_d)

        # 级联膨胀卷积+Swish激活
        self.conv_block = nn.Sequential(
            nn.Conv2d(mid_d, mid_d, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(mid_d),
            Swish(inplace=True),
            nn.Conv2d(mid_d, mid_d, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(mid_d),
            Swish(inplace=True)
        )

    def forward(self, x):
        context = self.coordatt(x)

        x_out = self.conv_block(context)

        return x_out



















