import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from einops import rearrange
from .modules import BottleNeckASPP, SwinBlock

import pdb

class DualpathTransformerBlock(BaseModule):
    def __init__(self,
                in_channels,
                channels,
                stride=1,
                norm_cfg=None,
                init_cfg=None,
                coeff_bias=True,
                aspp_drop=0.1,
                **kwargs):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.channels = channels
        self.stride = stride
        self.norm_cfg = norm_cfg
        self.kwargs = kwargs
        self.shift = (self.kwargs['layer_index'] % 2) == 1
        
        self.multihead_base_channel = 32
        self.num_heads = int(self.channels / self.multihead_base_channel)
        
        # build skip connection
        if self.stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, channels, kernel_size=1, stride=stride, bias=False),
                build_norm_layer(norm_cfg, channels)[1])
        else:
            self.downsample = nn.Identity()
        
        self.input_conv = nn.Sequential(
            nn.Conv3d(in_channels, channels, kernel_size=3, 
                padding=1, stride=stride, bias=False),
            build_norm_layer(norm_cfg, channels)[1],
            nn.ReLU(inplace=True),
        )
        
        # shared window attention
        self.bev_encoder = SwinBlock(
            embed_dims=self.channels,
            num_heads=self.num_heads,
            feedforward_channels=self.channels,
            window_size=7,
            drop_path_rate=0.2,
            shift=self.shift)
        
        # aspp in global path
        self.aspp = BottleNeckASPP(inplanes=self.channels, norm_cfg=self.norm_cfg, dropout=aspp_drop)
        
        # soft weights for fusion
        self.combine_coeff = nn.Conv3d(self.channels, 1, kernel_size=1, bias=coeff_bias)
        
    def forward(self, x):
        input_identity = x.clone()
        x = self.input_conv(x)
        
        x_bev = x.mean(dim=-1)
        batch_size = x_bev.shape[0]
        
        x = rearrange(x, 'b c x y z -> (b z) c x y')
        x = torch.cat((x_bev, x), dim=0)
        x = self.bev_encoder(x) # relu output
        x_bev, x = x[:batch_size], x[batch_size:] 
        x = rearrange(x, '(b z) c x y -> b c x y z', b=batch_size)
        x_bev = self.aspp(x_bev)
        
        coeff = self.combine_coeff(x).sigmoid()
        x = x + coeff * x_bev.unsqueeze(-1)
        
        return x + self.downsample(input_identity)