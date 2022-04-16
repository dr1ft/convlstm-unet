""" Full assembly of the parts to form the complete network """
import numpy as np
import torch

"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F
from model.Convlstm import ConvLSTM
from model.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # self.down4 = Down(512, 512)
        self.cvlstm1 = ConvLSTM(128, 128, [(3, 3)], 1, True, True, False)
        self.cvlstm2 = ConvLSTM(512, 512, [(3, 3)], 1, True, True, False)
        self.up1 = Up(768, 256, bilinear)
        self.up2 = Up(384, 128, bilinear)
        # self.up3 = Up(256, 64, bilinear)
        self.up3 = Up(192, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        b, _, _, _, _ = x.shape

        x1, x2, x3, x4 = [], [], [], []
        print('----------************------------\nup block:')

        for i in range(b):
            a = input[i, ...]
            print(f'--------第{i + 1}个batch----------')
            print(f'input shape:\t', a.shape)
            x1.append(self.inc(a))
            print('x1 shape:\t', x1[i].shape)
            x2.append(self.down1(x1[i]))
            x3.append(self.down2(x2[i]))
            x4.append(self.down3(x3[i]))
            # x5.append(self.down4(x4[i]))
            print('final  batch out x4[i] shape:\t', x4[i].shape)
        x1 = torch.stack(x1)
        x2 = torch.stack(x2)
        x3 = torch.stack(x3)
        x4 = torch.stack(x4)

        print('\nfinal up x4:\t', type(x4), x4.shape)
        print('----------************------------\nconvlstm block:')
        print('x4_target shape:\t', x4[:, -3:, ...].shape)

        x2_data, x2_target = x2[:, 0:3, ...], x2[:, -3:, ...]
        x2_cl_outs = self.cvlstm1(x2_data)
        x4_data, x4_target = x4[:, 0:3, ...], x4[:, -3:, ...]
        x4_cl_outs = self.cvlstm2(x4_data)
        x4 = x4_target
        b, _, _, _, _ = x4.shape
        logits = []
        print('----------************------------\ndown block:')
        for i in range(b):
            print(f'--------第{i + 1}个batch----------')
            print('input x4 and conv x4:\t', x4[i, ...].shape, x4_cl_outs[0][0][i,...].shape)
            x = self.up1(x4_cl_outs[0][0][i,...], x3[i, -3:, ...])
            print('after featrue cat with x3:\t', x.shape)
            x = self.up2(x, x2_cl_outs[0][0][i])
            print('after conv cat with x2:\t', x.shape)
            x = self.up3(x, x1[i, -3:, ...])
            print('after feature cat with x1:', x.shape)

            logits.append(self.outc(x))
        logits = torch.stack(logits)
        return logits


if __name__ == '__main__':
    net = UNet(n_channels=1, n_classes=1)
    input = torch.randn((2, 6, 1, 512, 512))
    logits = net(input)
    print('result:',logits.shape)
