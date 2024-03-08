import torch
from torch import nn as nn

from basicsr.archs.okdn_blocks import OKDB, BSConvU, BSConvU_idt, BSConvU_rep, UpsampleOneStep, Upsampler_rep, CCALayer,  ESA
from basicsr.utils.registry import ARCH_REGISTRY
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os


@ARCH_REGISTRY.register()
class OKDN(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=54,
                 num_atten=54,
                 num_block=8,
                 upscale=4,
                 num_in=4,
                 conv='BSConvU',
                 upsampler='pixelshuffledirect'):
        super().__init__()
        self.num_in = num_in
        if conv == 'BSConvU_idt':
            self.conv = BSConvU_idt
        elif conv == 'BSConvU_rep':
            self.conv = BSConvU_rep
        elif conv == 'BSConvU':
            self.conv = BSConvU
        else:
            raise NotImplementedError(f'conv {conv} is not supported yet.')
        print(conv)
        self.fea_conv = BSConvU_idt(num_in_ch * num_in, num_feat, kernel_size=3, padding=1)

        self.B1 = OKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B2 = OKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B3 = OKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B4 = OKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B5 = OKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B6 = OKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B7 = OKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B8 = OKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.esa = ESA(num_feat, conv=self.conv)
        self.cca = CCALayer(num_feat)
        self.GELU = nn.GELU()

        self.c2 = BSConvU_idt(num_feat, num_feat, kernel_size=3, padding=1)

        if upsampler == 'pixelshuffledirect':
            self.upsampler = UpsampleOneStep(num_feat, num_out_ch, upscale_factor=upscale)
        elif upsampler == 'pixelshuffle_rep':
            self.upsampler = Upsampler_rep(num_feat, num_out_ch, upscale_factor=upscale)
        else:
            raise NotImplementedError("Check the Upsampler. None or not support yet.")

    def forward(self, input):
        input = torch.cat([input] * self.num_in, dim=1)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1)
        out_B = self.c1(trunk)
        out_B = self.esa(out_B)
        out_B = self.cca(out_B)
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea
        output = self.upsampler(out_lr)

        return output
