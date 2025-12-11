import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from Mamba_backbone import Backbone_VSSM
from vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from ChangeDecoder import ChangeDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count


class STMambaBCD(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(STMambaBCD, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
 

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.main_clf = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # Decoder processing - passing encoder outputs to the decoder
        output = self.decoder(pre_features, post_features)

        output = self.main_clf(output)
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear')
        return output


if __name__ == "__main__":
    # Complete configuration with all required parameters
    model_kwargs = {
        'norm_layer': 'ln2d',
        'ssm_act_layer': 'silu',
        'mlp_act_layer': 'gelu',
        'depths': [2, 2, 9, 2],
        'embed_dim': 96,
        'ssm_d_state': 16,
        'ssm_ratio': 2.0,
        'ssm_dt_rank': "auto",
        'ssm_conv': 3,
        'ssm_conv_bias': True,
        'ssm_drop_rate': 0.0,
        'ssm_init': 'v0',
        'ssm_forwardtype': 'v2',  # This is the correct parameter name from config
        'mlp_ratio': 4.0,
        'mlp_drop_rate': 0.0,
        'patch_norm': True,
        'downsample': 'v2',
        'patchembed': 'v2',
        'gmlp': False,
        'use_checkpoint': False
    }

    # Prepare dummy inputs
    dummy_input1 = torch.randn(1, 3, 256, 256).cuda()
    dummy_input2 = torch.randn(1, 3, 256, 256).cuda()

    # Initialize model
    model = STMambaBCD(pretrained=None, **model_kwargs).cuda()

    # Test forward pass
    output = model(dummy_input1, dummy_input2)
    print("Output shape:", output.shape)