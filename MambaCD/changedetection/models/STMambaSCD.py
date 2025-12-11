import torch
from Mamba_backbone import Backbone_VSSM
from vmamba import LayerNorm2d
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from ChangeDecoder import ChangeDecoder
from SemanticDecoder import SemanticDecoder

class STMambaSCD_NoDownsample(nn.Module):
    def __init__(self, output_cd=2, output_clf=11, pretrained=None, **kwargs):
        super(STMambaSCD_NoDownsample, self).__init__()
        
        # 从 kwargs 中获取 dims，若未提供则使用默认值 [64, 64, 64, 64]
        dims = kwargs.get('dims', [64, 64, 64, 64])
        encoder_output_channels = dims[-1]  # 使用 dims 的最后一维作为通道数
        
        # 确保 kwargs 中包含 dims
        kwargs['dims'] = dims
        
        # 其他配置（如 downsample_version 和 patch_size）
        kwargs['downsample_version'] = kwargs.get('downsample_version', 'none')
        kwargs['patch_size'] = kwargs.get('patch_size', 1)
        
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
            output_channels=encoder_output_channels,
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

        self.main_clf_cd = nn.Conv2d(in_channels=128, out_channels=output_cd, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_clf, kernel_size=1)

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

        return output_bcd, output_T1, output_T2

if __name__ == "__main__":
    # 配置参数
    model_kwargs = {
        'norm_layer': 'ln2d',
        'ssm_act_layer': 'silu',
        'mlp_act_layer': 'gelu',
        'depths': [2, 2, 9, 2],  # 4个阶段的块数
        'dims': [96, 192, 384, 768],  # 每个阶段的通道数
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
        'downsample_version': 'v2',  # 禁用下采样
        'patchembed_version': 'v1',
        'gmlp': False,
        'use_checkpoint': False
    }

    # 创建测试输入
    dummy_input1 = torch.randn(1, 3, 128, 128).cuda()
    dummy_input2 = torch.randn(1, 3, 128, 128).cuda()

    # 创建模型
    model = STMambaSCD_NoDownsample(**model_kwargs).cuda()
    
    # 前向传播
    cd_out, t1_out, t2_out = model(dummy_input1, dummy_input2)
    
    # 打印输出形状
    print(f"变化检测输出形状: {cd_out.shape}")  # 应该是 [1, 2, 256, 256]
    print(f"时相1语义分割输出形状: {t1_out.shape}")  # 应该是 [1, 11, 256, 256]
    print(f"时相2语义分割输出形状: {t2_out.shape}")  # 应该是 [1, 11, 256, 256]
    print(f"时相2语义分割输出形状: {t2_out.shape}")  # 应该是 [1, 11, 256, 256]
    print(f"时相2语义分割输出形状: {t2_out.shape}")  # 应该是 [1, 11, 256, 256]