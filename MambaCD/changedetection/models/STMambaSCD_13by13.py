import torch
from .Mamba_backbone import Backbone_VSSM
from .vmamba import LayerNorm2d
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from .ChangeDecoder import ChangeDecoder
from .SemanticDecoder import SemanticDecoder

class STMambaSCD_NoDownsample(nn.Module):
    def __init__(self, output_cd=2, output_clf=11, pretrained=None, **kwargs):
        super(STMambaSCD_NoDownsample, self).__init__()
        
        # 修改后的encoder配置，没有下采样
        kwargs['depths'] = [1, 1, 1, 1]  # 减少每个阶段的块数
        kwargs['dims'] = [64, 64, 64, 64]  # 保持相同的通道数
        kwargs['patch_size'] = 1  # 避免初始patch嵌入的下采样
        kwargs['downsample_version'] = 'none'  # 禁用下采样
        kwargs['in_chans'] = 138  # 设置输入通道数为138
        
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
        
        # 修改decoder配置以适应无下采样的特征
        self.decoder_bcd = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            output_channels=64,  # 使用固定的输出通道数
            **clean_kwargs
        )

        self.decoder_T1 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_T2 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        # 使用固定的输入通道数
        self.main_clf_cd = nn.Conv2d(in_channels=128, out_channels=output_cd, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_clf, kernel_size=1)

        # 添加分类头
        self.classifier_cd = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 将空间维度压缩为1x1
            nn.Flatten(),  # 将特征图展平为向量
            nn.Linear(output_cd, 2)  # 全连接层，输出类别数
        )

        self.classifier_T1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(output_clf, output_clf)
        )

        self.classifier_T2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(output_clf, output_clf)
        )

    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # Decoder processing
        output_bcd = self.decoder_bcd(pre_features, post_features)
        output_T1 = self.decoder_T1(pre_features)
        output_T2 = self.decoder_T2(post_features)

        # 直接卷积输出，不需要上采样
        output_bcd = self.main_clf_cd(output_bcd)
        output_T1 = self.aux_clf(output_T1)
        output_T2 = self.aux_clf(output_T2)

        # 分类输出
        cls_cd = self.classifier_cd(output_bcd)  # 变化检测的分类输出
        cls_T1 = self.classifier_T1(output_T1)  # 时相1的分类输出
        cls_T2 = self.classifier_T2(output_T2)  # 时相2的分类输出

        return {
            'change': cls_cd,  # 变化检测的分类输出
            'class_2010': cls_T1,  # 时相1的分类输出
            'class_2015': cls_T2   # 时相2的分类输出
        }

if __name__ == "__main__":
    # 配置参数
    model_kwargs = {
        'norm_layer': 'ln2d',
        'ssm_act_layer': 'silu',
        'mlp_act_layer': 'gelu',
        'depths': [1, 1, 1, 1],  # 减少每个阶段的块数
        'dims': [64, 64, 64, 64],  # 保持相同的通道数
        'ssm_d_state': 16,
        'ssm_ratio': 2.0,
        'ssm_dt_rank': "auto",
        'ssm_conv': 3,
        'ssm_conv_bias': True,
        'ssm_drop_rate': 0.0,
        'ssm_init': 'v0',
        'forward_type': 'v2',
        'mlp_ratio': 4.0,
        'mlp_drop_rate': 0.0,
        'patch_norm': True,
        'downsample_version': 'none',  # 禁用下采样
        'patchembed_version': 'v1',
        'gmlp': False,
        'use_checkpoint': False,
        'in_chans': 138  # 设置输入通道数为138
    }

    # 创建测试输入
    dummy_input1 = torch.randn(1, 138, 13, 13).cuda()  # 修改输入通道数为138
    dummy_input2 = torch.randn(1, 138, 13, 13).cuda()  # 修改输入通道数为138

    # 创建模型
    model = STMambaSCD_NoDownsample(**model_kwargs).cuda()
    
    # 前向传播
    outputs = model(dummy_input1, dummy_input2)
    
    # 打印输出形状
    print(f"变化检测输出形状: {outputs['change'].shape}")  # 应该是 [1, 2, 13, 13]
    print(f"时相1语义分割输出形状: {outputs['class_2010'].shape}")  # 应该是 [1, 11, 13, 13]
    print(f"时相2语义分割输出形状: {outputs['class_2015'].shape}")  # 应该是 [1, 11, 13, 13]
    print(f"变化检测分类输出形状: {outputs['classification_cd'].shape}")  # 应该是 [1, 11]
    print(f"时相1分类输出形状: {outputs['classification_T1'].shape}")  # 应该是 [1, 11]
    print(f"时相2分类输出形状: {outputs['classification_T2'].shape}")  # 应该是 [1, 11]