from abc import ABCMeta

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from mmcv.cnn import ConvModule


def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class BaseDecodeHead(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_channels, channels, *, num_classes, dropout_ratio=0.1, conv_cfg=None, norm_cfg=None,
                 act_cfg=dict(type='SiLU'), in_index=-1, input_transform=None, ignore_index=255,
                 align_corners=False):
        super(BaseDecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
            

    def _init_inputs(self, in_channels, in_index, input_transform):
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [resize(input=x, size=inputs[0].shape[2:], mode='bilinear', 
                                       align_corners=self.align_corners) for x in inputs]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def cls_seg(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output


class PPM(nn.ModuleList):
    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg, act_cfg, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_scale),
                ConvModule(self.in_channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
                           act_cfg=self.act_cfg, inplace=True)))

    def forward(self, x):
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(ppm_out, size=x.size()[2:], mode='bilinear', align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

    
class UPerHead(BaseDecodeHead):
    def __init__(self, pool_scales=(2, 3, 4, 6), **kwargs):
        super(UPerHead, self).__init__(input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(pool_scales, self.in_channels[-1], self.channels, conv_cfg=self.conv_cfg,
                               norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, align_corners=self.align_corners)
        
        self.bottleneck = ConvModule(self.in_channels[-1] + len(pool_scales) * self.channels, self.channels, 3,
                                     padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(in_channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg, inplace=False)
            fpn_conv = ConvModule(self.channels, self.channels, 3, padding=1, conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(len(self.in_channels) * self.channels, self.channels, 3, padding=1,
                                         conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False)

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        
        # build laterals
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        laterals.append(self.psp_forward(inputs))
        
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(laterals[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
        
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(fpn_outs[i], size=fpn_outs[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
        
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(output)
        return output
    
    
class FCNHead(BaseDecodeHead):
    def __init__(self, num_convs=2, kernel_size=3, concat_input=True, **kwargs):
        assert num_convs >= 0
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        convs = []
        convs.append(
            ConvModule(self.in_channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2,
                       conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(ConvModule(self.channels, self.channels, kernel_size=kernel_size,
                                    padding=kernel_size // 2,conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(self.in_channels + self.channels, self.channels, kernel_size=kernel_size,
                                       padding=kernel_size // 2, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
