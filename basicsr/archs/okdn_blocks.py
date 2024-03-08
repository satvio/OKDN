import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from basicsr.archs.arch_util import default_init_weights
from basicsr.archs import SwinT

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import numpy as np
import os
import math




class ShiftConv2d_4(nn.Module):
    def __init__(self, inp_channels, move_channels=2, move_pixels=2):
        super(ShiftConv2d_4, self).__init__()
        self.inp_channels = inp_channels
        self.move_p = move_pixels
        self.move_c = move_channels
        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        mid_channel = inp_channels // 2
        up_channels = (mid_channel - move_channels * 2, mid_channel - move_channels)
        down_channels = (mid_channel - move_channels, mid_channel)
        left_channels = (mid_channel, mid_channel + move_channels)
        right_channels = (mid_channel + move_channels, mid_channel + move_channels * 2)
        self.weight[left_channels[0]:left_channels[1], 0, 1, 2] = 1.0  ## left
        self.weight[right_channels[0]:right_channels[1], 0, 1, 0] = 1.0  ## right
        self.weight[up_channels[0]:up_channels[1], 0, 2, 1] = 1.0  ## up
        self.weight[down_channels[0]:down_channels[1], 0, 0, 1] = 1.0  ## down
        self.weight[0:mid_channel - move_channels * 2, 0, 1, 1] = 1.0  ## identity
        self.weight[mid_channel + move_channels * 2:, 0, 1, 1] = 1.0  ## identity

    def forward(self, x):
        for i in range(self.move_p):
            x = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)

        return x


# LayerNorm
## Taken from ConvNeXt: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()

        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x






class BSConvU(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True,
                 padding_mode="zeros"):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea



class BSConvU_idt(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(BSConvU_idt, self).__init__()
        self.in_planes = in_channels
        self.out_planes = out_channels
        self.kernel_size = kernel_size

        self.point_wise = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=True)

        # self.point_wise_bias = nn.Conv2d(in_channels=out_channels,
        #                             #      in_channels=in_channels,
        #                             out_channels=out_channels,
        #                             kernel_size=1,
        #                             stride=1,
        #                             padding=0,
        #                             groups=1,
        #                             bias=True)

        # self.weight =  nn.Parameter(torch.normal(mean=0, std=0.0001, size=(out_channels, out_channels, 1, 1)))
        self.weight = nn.Parameter(torch.normal(mean=0, std=0.0001, size=(out_channels, 1, kernel_size, kernel_size)))
        self.weight_2 = nn.Parameter(torch.normal(mean=0, std=0.0001, size=(out_channels, out_channels, 1, 1)))
        # self.weight = torch.normal(mean=0, std=0.0005, size=(out_channels, 1, kernel_size, kernel_size))
        # self.weight = nn.Parameter(self.weight.repeat(1, out_channels, 1, 1))
        self.depth_wise = nn.Conv2d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding,
                                    groups=out_channels,
                                    bias=True)

    def forward(self, x):  #
        # print('x',x.size())
        # print('input',x.size())
        Blue_tmp = self.point_wise(x)  #
        # print('Blue_tmp_point', Blue_tmp.size())
        Blue_tmp = self.depth_wise(Blue_tmp)  #

        bias = F.conv2d(Blue_tmp, self.weight, padding=(self.kernel_size - 1) // 2, groups=self.out_planes)
        bias = F.conv2d(bias, self.weight_2)
        # print('bias:', bias.size())

        out = Blue_tmp + bias

        return out


class BSConvU_rep(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=True):
        super(BSConvU_rep, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_bias

        # Generating local adaptive weights
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_size ** 2, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size ** 2, kernel_size ** 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size ** 2, kernel_size ** 2, 1),
            nn.Sigmoid()
        )  # b,9,H,W È«Í¨µÀÏñËØ¼¶ºË×¢ÒâÁ¦
        # self.attention2=nn.Sequential(
        #     nn.Conv2d(in_planes,(kernel_size**2)*in_planes,kernel_size, stride, padding,groups=in_planes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d((kernel_size**2)*in_planes,(kernel_size**2)*in_planes,1,groups=in_planes),
        #     nn.Sigmoid()
        # ) #b,9n,H,W µ¥Í¨µÀÏñËØ¼¶ºË×¢ÒâÁ¦
        if use_bias == True:  # Global local adaptive weights
            self.attention3 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_planes, out_planes, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_planes, out_planes, 1)
            )  # b,m,1,1 Í¨µÀÆ«ÖÃ×¢ÒâÁ¦

        conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups)
        self.weight = conv1.weight  # m, n, k, k

    def forward(self, x):
        (b, n, H, W) = x.shape
        m = self.out_planes
        k = self.kernel_size
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)
        atw1 = self.attention1(x)  # b,k*k,n_H,n_W
        # atw2=self.attention2(x) #b,n*k*k,n_H,n_W

        atw1 = atw1.permute([0, 2, 3, 1])  # b,n_H,n_W,k*k
        atw1 = atw1.unsqueeze(3).repeat([1, 1, 1, n, 1])  # b,n_H,n_W,n,k*k
        atw1 = atw1.view(b, n_H, n_W, n * k * k)  # b,n_H,n_W,n*k*k

        # atw2=atw2.permute([0,2,3,1]) #b,n_H,n_W,n*k*k

        atw = atw1  # *atw2 #b,n_H,n_W,n*k*k
        atw = atw.view(b, n_H * n_W, n * k * k)  # b,n_H*n_W,n*k*k
        atw = atw.permute([0, 2, 1])  # b,n*k*k,n_H*n_W

        kx = F.unfold(x, kernel_size=k, stride=self.stride, padding=self.padding)  # b,n*k*k,n_H*n_W
        atx = atw * kx  # b,n*k*k,n_H*n_W

        atx = atx.permute([0, 2, 1])  # b,n_H*n_W,n*k*k
        atx = atx.view(1, b * n_H * n_W, n * k * k)  # 1,b*n_H*n_W,n*k*k

        w = self.weight.view(m, n * k * k)  # m,n*k*k
        w = w.permute([1, 0])  # n*k*k,m
        y = torch.matmul(atx, w)  # 1,b*n_H*n_W,m
        y = y.view(b, n_H * n_W, m)  # b,n_H*n_W,m
        if self.bias == True:
            bias = self.attention3(x)  # b,m,1,1
            bias = bias.view(b, m).unsqueeze(1)  # b,1,m
            bias = bias.repeat([1, n_H * n_W, 1])  # b,n_H*n_W,m
            y = y + bias  # b,n_H*n_W,m

        y = y.permute([0, 2, 1])  # b,m,n_H*n_W
        y = F.fold(y, output_size=(n_H, n_W), kernel_size=1)  # b,m,n_H,n_W
        return y


class Attention(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

    def forward(self, x):
        u = x.clone()
        attn = self.pointwise(x)
        attn = self.depthwise(attn)
        attn = self.depthwise_dilated(attn)
        return u * attn


class LSKA(nn.Module):
    def __init__(self, dim, k_size):
        super().__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, 2), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), groups=dim,
                                            dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, 4), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), groups=dim,
                                            dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1, 1), padding=(0, 9), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1, 1), padding=(9, 0), groups=dim,
                                            dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1, 1), padding=(0, 15), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1, 1), padding=(15, 0), groups=dim,
                                            dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1, 1), padding=(0, 18), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1, 1), padding=(18, 0), groups=dim,
                                            dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1, 1), padding=(0, 24), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1, 1), padding=(24, 0), groups=dim,
                                            dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn


class SA(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(SA, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out


class SI(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SI, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ESA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d, p=0.25):
        super(ESA, self).__init__()
        f = num_feat // 4
        BSConvS_kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': p}
        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv2 = conv(f, f, 3, 2, 0)
        self.conv3 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_ = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        c1_ = (self.conv1(input))
        c1 = self.conv2(c1_)
        v_max = self.maxPooling(c1)
        v_range = self.GELU(self.conv_max(v_max))
        c3 = self.GELU(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4((c3 + cf))
        m = self.sigmoid(c4)

        return input * m


class OKDB(nn.Module):

    def __init__(self, in_channels, out_channels, atten_channels=None, conv=nn.Conv2d):
        super().__init__()

        self.dc = self.distilled_channels = in_channels // 3
        self.rc = self.remaining_channels = in_channels
        if (atten_channels is None):
            self.atten_channels = in_channels
        else:
            self.atten_channels = atten_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3, padding=1)
        self.c2_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c2_r = conv(self.rc, self.rc, kernel_size=3, padding=1)
        self.c3_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c3_r = conv(self.rc, self.rc, kernel_size=3, padding=1)
        self.c4_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c4_r = conv(self.rc, self.rc, kernel_size=3, padding=1)

        self.c5 = BSConvU(self.rc, self.dc, kernel_size=3, padding=1)
        self.act = nn.GELU()

        self.c6 = nn.Conv2d(self.dc * 5, self.atten_channels, 1)
        self.atten = LSKA(self.atten_channels, 35)
        self.c7 = nn.Conv2d(self.atten_channels, out_channels, 1)
        self.pixel_norm = nn.LayerNorm(out_channels)
        default_init_weights([self.pixel_norm], 0.1)
        self.sf = ShiftConv2d_4(in_channels)

    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input))

        r_c1 = (self.c1_r(self.sf(input)))
        r_c1 = self.act(r_c1)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(self.sf(r_c1)))
        r_c2 = self.act(r_c2)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(self.sf(r_c2)))
        r_c3 = self.act(r_c3)

        distilled_c4 = self.act(self.c4_d(r_c3))
        r_c4 = (self.c4_r(self.sf(r_c3)))
        r_c4 = self.act(r_c4)

        r_c5 = self.act(self.c5(r_c4))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4, r_c5], dim=1)
        out = self.c6(out)

        out_fused = self.atten(out)
        out_fused = self.c7(out_fused)
        out_fused = out_fused.permute(0, 2, 3, 1)
        out_fused = self.pixel_norm(out_fused)
        out_fused = out_fused.permute(0, 3, 1, 2).contiguous()

        return out_fused + input


def UpsampleOneStep(in_channels, out_channels, upscale_factor=4):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


class Upsampler_rep(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_factor=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), 3, 1, 1)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels * 2, 1)
        self.conv3x3 = nn.Conv2d(in_channels * 2, out_channels * (upscale_factor ** 2), 3)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        v1 = F.conv2d(x, self.conv1x1.weight, self.conv1x1.bias, padding=0)
        v1 = F.pad(v1, (1, 1, 1, 1), 'constant', 0)
        b0_pad = self.conv1x1.bias.view(1, -1, 1, 1)
        v1[:, :, 0:1, :] = b0_pad
        v1[:, :, -1:, :] = b0_pad
        v1[:, :, :, 0:1] = b0_pad
        v1[:, :, :, -1:] = b0_pad
        v2 = F.conv2d(v1, self.conv3x3.weight, self.conv3x3.bias, padding=0)
        out = self.conv1(x) + self.conv3(x) + v2
        return self.pixel_shuffle(out)
