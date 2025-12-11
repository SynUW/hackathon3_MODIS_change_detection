"""
https://github.com/ChenHongruixuan/ChangeMamba/blob/master/changedetection/models/STMambaSCD.py
"""
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from ChangeDecoder import ChangeDecoder
from SemanticDecoder import SemanticDecoder

class STMambaSCD(nn.Module):
    def __init__(self, output_cd, output_clf, pretrained,  **kwargs):
        super(STMambaSCD, self).__init__()
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

        self.channel_first = self.encoder.channel_first

        print(self.channel_first)

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)


        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder_bcd = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_T1 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_T2 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )


        self.main_clf_cd = nn.Conv2d(in_channels=128, out_channels=output_cd, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_clf, kernel_size=1)


    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # Decoder processing - passing encoder outputs to the decoder
        output_bcd = self.decoder_bcd(pre_features, post_features)
        output_T1 = self.decoder_T1(pre_features)
        output_T2 = self.decoder_T2(post_features)


        output_bcd = self.main_clf_cd(output_bcd)
        output_bcd = F.interpolate(output_bcd, size=pre_data.size()[-2:], mode='bilinear')

        output_T1 = self.aux_clf(output_T1)
        output_T1 = F.interpolate(output_T1, size=pre_data.size()[-2:], mode='bilinear')
        
        output_T2 = self.aux_clf(output_T2)
        output_T2 = F.interpolate(output_T2, size=post_data.size()[-2:], mode='bilinear')

        return output_bcd, output_T1, output_T2

if __name__ == '__main__':
    # 测试配置
    config = {
        'output_cd': 1,  # 变化检测输出通道数
        'output_clf': 7,  # 语义分割类别数
        'pretrained': False,  # 是否使用预训练模型
        'norm_layer': 'ln2d',  # 归一化层类型
        'ssm_act_layer': 'silu',  # SSM激活函数
        'mlp_act_layer': 'gelu',  # MLP激活函数
        'drop_path_rate': 0.1,  # Drop path率
        'use_checkpoint': False,  # 是否使用checkpoint
        'channel_first': False,  # 通道优先
        'depths': [2, 2, 9, 2],  # 各阶段深度
        'dims': [96, 192, 384, 768],  # 各阶段维度
        'ssm_d_state': 16,  # SSM状态维度
        'ssm_dt_rank': 'auto',  # SSM dt rank
        'ssm_ratio': 2,  # SSM比率
        'ssm_conv': 3,  # SSM卷积核大小
        'ssm_conv_bias': True,  # SSM卷积是否使用偏置
        'ssm_drop_rate': 0.0,  # SSM dropout率
        'ssm_init': 'v0',  # SSM初始化方式
        'forward_type': 'v2',  # 前向传播类型
        'mlp_ratio': 4.0,  # MLP比率
        'mlp_drop_rate': 0.0,  # MLP dropout率
        'gmlp': False,  # 是否使用gMLP
        'drop_rate': 0.0,  # Dropout率
        'attn_drop_rate': 0.0,  # 注意力Dropout率
        'img_size': 224,  # 图像大小
        'patch_size': 4,  # 图像块大小
        'in_chans': 3,  # 输入通道数
    }
    
    # 创建模型实例
    model = STMambaSCD(**config)
    
    # 打印模型架构
    print("模型架构:")
    print(model)
    
    # 测试前向传播
    batch_size = 2
    channels = 3
    height, width = 224, 224  # 使用模型默认的输入尺寸
    
    # 创建测试输入
    pre_data = torch.randn(batch_size, channels, height, width)
    post_data = torch.randn(batch_size, channels, height, width)
    
    # 训练模式测试
    print("\n训练模式测试:")
    model.train()
    output_bcd, output_T1, output_T2 = model(pre_data, post_data)
    print(f"变化检测输出 shape: {output_bcd.shape}")
    print(f"时相1语义分割 shape: {output_T1.shape}")
    print(f"时相2语义分割 shape: {output_T2.shape}")
    
    # 推理模式测试
    print("\n推理模式测试:")
    model.eval()
    with torch.no_grad():
        output_bcd, output_T1, output_T2 = model(pre_data, post_data)
    print(f"变化检测输出 shape: {output_bcd.shape}")
    print(f"时相1语义分割 shape: {output_T1.shape}")
    print(f"时相2语义分割 shape: {output_T2.shape}")
    
    # 测试模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 测试模型在不同设备上的运行
    if torch.cuda.is_available():
        print("\nGPU测试:")
        device = torch.device('cuda')
        model = model.to(device)
        pre_data = pre_data.to(device)
        post_data = post_data.to(device)
        with torch.no_grad():
            output_bcd, output_T1, output_T2 = model(pre_data, post_data)
        print("GPU运行成功!")
        
    # 测试不同输入尺寸
    print("\n不同输入尺寸测试:")
    test_sizes = [(128, 128), (224, 224), (256, 256), (384, 384)]
    for h, w in test_sizes:
        try:
            pre_data = torch.randn(batch_size, channels, h, w)
            post_data = torch.randn(batch_size, channels, h, w)
            if torch.cuda.is_available():
                pre_data = pre_data.to(device)
                post_data = post_data.to(device)
            with torch.no_grad():
                output_bcd, output_T1, output_T2 = model(pre_data, post_data)
            print(f"输入尺寸 {h}x{w} 测试成功!")
        except Exception as e:
            print(f"输入尺寸 {h}x{w} 测试失败: {str(e)}")
            
    # 测试内存使用
    print("\n内存使用测试:")
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        print(f"CPU内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        if torch.cuda.is_available():
            print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
    except ImportError:
        print("psutil未安装，无法测试内存使用")
        
    # 测试模型推理时间
    print("\n推理时间测试:")
    import time
    num_runs = 100
    model.eval()
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = model(pre_data, post_data)
        
        # 计时
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(pre_data, post_data)
        end_time = time.time()
        
    avg_time = (end_time - start_time) / num_runs * 1000  # 转换为毫秒
    print(f"平均推理时间: {avg_time:.2f} ms")