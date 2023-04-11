import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from copy import deepcopy

class _ASPPModule(BaseModule):
    def __init__(self, 
            inplanes, 
            planes, 
            kernel_size, 
            padding, 
            dilation,
            norm_cfg=dict(type='BN'),
            conv_cfg=None,
        ):
        super(_ASPPModule, self).__init__()
        
        self.atrous_conv = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.bn(self.atrous_conv(x))
        x = self.relu(x)

        return x

class ASPP(BaseModule):
    def __init__(self,
            inplanes,
            mid_channels=None,
            dilations=[1, 6, 12, 18],
            norm_cfg=dict(type='BN'),
            conv_cfg=None,
            dropout=0.1,
        ):
        super(ASPP, self).__init__()
        
        if mid_channels is None:
            mid_channels = inplanes // 2
        
        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 norm_cfg=norm_cfg)
        
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 norm_cfg=norm_cfg)
        
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 norm_cfg=norm_cfg)
        
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 norm_cfg=norm_cfg)
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            build_conv_layer(conv_cfg, inplanes, mid_channels, 1, stride=1, bias=False),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(inplace=True),
        )
        
        # we set the output channel the same as the input
        outplanes = inplanes
        self.conv1 = build_conv_layer(conv_cfg, int(mid_channels * 5), outplanes, 1, bias=False)
        self.bn1 = build_norm_layer(norm_cfg, outplanes)[1]
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weight()

    def forward(self, x):
        identity = x.clone()
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        return identity + self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
class BottleNeckASPP(BaseModule):
    def __init__(self,
            inplanes,
            reduction=4,
            dilations=[1, 6, 12, 18],
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            conv_cfg=None,
            dropout=0.1,
        ):
        super(BottleNeckASPP, self).__init__()
        
        channels = inplanes // reduction
        self.input_conv = nn.Sequential(
            build_conv_layer(conv_cfg, inplanes, channels, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, channels)[1],
            nn.ReLU(inplace=True),
        )
        
        assert norm_cfg['type'] == 'GN'
        # when num_group >= num_channel because of the reduction, reduce the num_group
        aspp_norm_cfg = deepcopy(norm_cfg)
        if channels <= norm_cfg['num_groups']:
            aspp_norm_cfg['num_groups'] = channels // 2
        
        # aspp_norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)
        self.aspp = ASPP(channels, mid_channels=channels, dropout=dropout,
                dilations=dilations, norm_cfg=aspp_norm_cfg)
        
        self.output_conv = nn.Sequential(
            build_conv_layer(conv_cfg, channels, inplanes, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, inplanes)[1],
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        identity = x
        x = self.input_conv(x)
        x = self.aspp(x)
        x = self.output_conv(x)
        
        return identity + x