import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import logging
from scipy import ndimage

from lib.attention import AttentionModule
from lib.decoders import CASCADE
from lib.pvtv2 import pvt_v2_b2
from lib.segformer import M_EfficientSelfAtten, LE_FFN, DWConv

logger = logging.getLogger(__name__)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class MSI_LE_Layer(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = LE_FFN(dims, dims * 4)
        self.mixffn2 = LE_FFN(dims * 2, dims * 8)
        self.mixffn3 = LE_FFN(dims * 5, dims * 20)
        self.mixffn4 = LE_FFN(dims * 8, dims * 32)
        self.AM1= AttentionModule(dims)
        self.AM2 = AttentionModule(dims * 2)
        self.AM3 = AttentionModule(dims * 5)
        self.AM4 = AttentionModule(dims* 8)

    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3, c4 = inputs

            B, C, _, _ = c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64

            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)
        else:
            B, _, C = inputs.shape


        inputs_low = inputs[:, :4704, :]
        inputs_high = inputs[:, 4704:, :]


        tx_low = inputs_low + self.attn(self.norm1(inputs_low))
        # print("low==============")
        tx_high = inputs_high + self.attn(self.norm1(inputs_high))
        # print("high+++++++++")
        inputs = torch.cat([tx_low, tx_high], -2)

        tx1 = inputs + self.attn(self.norm1(inputs))


        tx = self.norm2(tx1)


        tem1 = tx[:, :3136, :].reshape(B, 56,56, C).permute(0, 3, 1, 2)
        tem2 = tx[:, 3136:4704, :].reshape(B, 28,28, C * 2).permute(0, 3, 1, 2)
        tem3 = tx[:, 4704:5684, :].reshape(B, 14,14, C * 5).permute(0, 3, 1, 2)
        tem4 = tx[:, 5684:6076, :].reshape(B, 7,7, C * 8).permute(0, 3, 1, 2)

        c1a = self.AM1(tem1)
        c2a = self.AM2(tem2)
        c3a = self.AM3(tem3)
        c4a = self.AM4(tem4)
        # print(c1a.shape)  torch.Size([1, 64, 56, 56])
        c1a = c1a.permute(0, 2, 3, 1).reshape(B, -1, C)
        c2a = c2a.permute(0, 2, 3, 1).reshape(B, -1, C * 2)
        c3a = c3a.permute(0, 2, 3, 1).reshape(B, -1, C * 5)
        c4a = c4a.permute(0, 2, 3, 1).reshape(B, -1, C* 8)


        m1f = self.mixffn1(c1a, 56, 56).reshape(B, -1, C)
        # print("2")
        m2f = self.mixffn2(c2a, 28, 28).reshape(B, -1, C)
        m3f = self.mixffn3(c3a, 14, 14).reshape(B, -1, C)
        m4f = self.mixffn4(c4a, 7, 7).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f, m3f, m4f], -2)
        # print(tx1.shape,t1.shape)
        tx2 = tx1 + t1


        return tx2
class MSI_LE_Block(nn.Module):
    def __init__(self, dims, head, reduction_ratios,connect_way):
        super().__init__()
        self.bridge_layer1 = MSI_LE_Layer(dims, head, reduction_ratios)
        self.bridge_layer2 = MSI_LE_Layer(dims, head, reduction_ratios)
        self.bridge_layer3 = MSI_LE_Layer(dims, head, reduction_ratios)
        self.bridge_layer4 = MSI_LE_Layer(dims, head, reduction_ratios)
        self.connection = connect_way

    def convert(self,x:torch.Tensor) -> torch.Tensor:
        if (type(x) == list):
            # print("-----1-----")
            c1, c2, c3, c4 = x
            B, C, _, _ = c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64

            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            x = torch.cat([c1f, c2f, c3f, c4f], -2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.connection == 'add':
            # print("add")
            bridge1 = self.bridge_layer1(x)
            bridge2 = self.bridge_layer2(bridge1)
            bridge3 = self.bridge_layer3(bridge2)
            bridge4 = self.bridge_layer4(bridge3)
            bridge4 = bridge1 + bridge2 + bridge3 + bridge4
        elif self.connection == 'residual':
            # print("residual")
            x_concat = self.convert(x)
            bridge1 = x_concat + self.bridge_layer1(x)
            bridge2 = bridge1 + self.bridge_layer2(bridge1)
            bridge3 = bridge2 + self.bridge_layer3(bridge2)
            bridge4 = bridge3 + self.bridge_layer4(bridge3)
        elif self.connection == 'dense_simple':
            # print("dense_simple")
            x_concat = self.convert(x)
            f = x_concat
            bridge1 = f + self.bridge_layer1(x)
            bridge2 = f + self.bridge_layer2(bridge1)
            bridge3 = f + self.bridge_layer3(bridge2)
            bridge4 = f + self.bridge_layer4(bridge3)
        else:
            # print("none")
            bridge1 = self.bridge_layer1(x)
            bridge2 = self.bridge_layer2(bridge1)
            bridge3 = self.bridge_layer3(bridge2)
            bridge4 = self.bridge_layer4(bridge3)


        B, _, C = bridge4.shape
        outs = []

        sk1 = bridge4[:, :3136, :].reshape(B, 56, 56, C).permute(0, 3, 1, 2)
        sk2 = bridge4[:, 3136:4704, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
        sk3 = bridge4[:, 4704:5684, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
        sk4 = bridge4[:, 5684:6076, :].reshape(B, 7, 7, C * 8).permute(0, 3, 1, 2)

        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs



class PVT_MSI_LE_CASCADE(nn.Module):
    def __init__(self, n_class=1,connect_way = 'none'):
        super(PVT_MSI_LE_CASCADE,self).__init__()

        # conv block to convert single channel to 3 channels
        # 把单通道转换为3通道
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.connect_way = connect_way
        self.reduction_ratios = [1, 2, 4, 8]
        self.bridge = MSI_LE_Block(64, 1,self.reduction_ratios,connect_way)

        # decoder initialization
        self.decoder = CASCADE(channels=[512, 320, 128, 64])

        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(512, n_class, 1)
        self.out_head2 = nn.Conv2d(320, n_class, 1)
        self.out_head3 = nn.Conv2d(128, n_class, 1)
        self.out_head4 = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        # print(x.shape)
        # if grayscale input, convert to 3 channels

        if x.size()[1] == 1:
            x = self.conv(x)


        encoder = self.backbone(x)

        bridge = self.bridge(encoder)  # list

        x1_o, x2_o, x3_o, x4_o = self.decoder(bridge[3], [bridge[2], bridge[1], bridge[0]])

        # prediction heads
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)

        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')
        return p1, p2, p3, p4


if __name__ == '__main__':
    model = PVT_MSI_LE_CASCADE(n_class=9,connect_way='dense_simple').cuda()
    input_tensor = torch.randn(1, 3, 224, 224).cuda()

    p1, p2, p3, p4 = model(input_tensor)
    print(p1.size(), p2.size(), p3.size(), p4.size())

