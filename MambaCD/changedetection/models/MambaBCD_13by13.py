import torch
import torch.nn as nn
import torch.nn.functional as F
from Mamba_backbone import Backbone_VSSM
from vmamba import VSSM, LayerNorm2d, VSSBlock, Permute


class STMambaBCD_NoDownsample(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(STMambaBCD_NoDownsample, self).__init__()

        # 完全禁用下采样的配置
        no_downsample_kwargs = kwargs.copy()
        no_downsample_kwargs['depths'] = [1]  # 只使用一个阶段
        no_downsample_kwargs['downsample'] = 'none'
        no_downsample_kwargs['patchembed'] = 'none'  # 禁用patch embedding

        # 修改encoder配置，确保不进行任何下采样
        self.encoder = Backbone_VSSM(
            out_indices=(0,),
            pretrained=pretrained,
            **no_downsample_kwargs
        )

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

        norm_layer = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)
        ssm_act_layer = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        # 简化decoder结构
        self.decoder = nn.Sequential(
            nn.Conv2d(self.encoder.dims[0] * 2, 128, kernel_size=3, padding=1),
            norm_layer(128) if norm_layer is not None else nn.Identity(),
            _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), nn.ReLU)(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            norm_layer(64) if norm_layer is not None else nn.Identity(),
            _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), nn.ReLU)(),
        )

        # self.main_clf = nn.Sequential(
        #     nn.Conv2d(64, 2, kernel_size=1),
        # )

        self.main_clf = nn.Sequential(
            # 全连接层 (B, C) -> (B, num_classes)
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, kwargs['num_classes']),
        )

    def forward(self, pre_data, post_data):
        # Encoder处理 - 不进行下采样
        pre_features = self.encoder(pre_data)[0]
        post_features = self.encoder(post_data)[0]

        # 合并特征
        combined = torch.cat([pre_features, post_features], dim=1)

        # Decoder处理
        output = self.decoder(combined)
        output = nn.AdaptiveAvgPool2d(1)(output)
        output = output.view(output.size(0), -1)
        # 最终分类
        output = self.main_clf(output)

        return output


if __name__ == "__main__":
    # 配置参数 - 确保不进行任何下采样
    model_kwargs = {
        'norm_layer': 'ln2d',
        'ssm_act_layer': 'silu',
        'mlp_act_layer': 'gelu',
        'depths': [1],  # 只使用一个阶段
        'embed_dim': 96,
        'ssm_d_state': 16,
        'ssm_ratio': 2.0,
        'ssm_dt_rank': "auto",
        'ssm_conv': 3,
        'ssm_conv_bias': True,
        'ssm_drop_rate': 0.0,
        'ssm_init': 'v0',
        'ssm_forwardtype': 'v2',
        'mlp_ratio': 4.0,
        'mlp_drop_rate': 0.0,
        'patch_norm': True,
        'downsample': 'none',  # 禁用下采样
        'patchembed': 'none',  # 禁用patch embedding
        'gmlp': False,
        'use_checkpoint': False,
        'num_classes': 11
    }

    # 准备输入数据
    dummy_input1 = torch.randn(1, 3, 13, 13).cuda()
    dummy_input2 = torch.randn(1, 3, 13, 13).cuda()

    # 初始化模型
    model = STMambaBCD_NoDownsample(pretrained=None, **model_kwargs).cuda()

    # 测试前向传播
    output = model(dummy_input1, dummy_input2)
    print("Output shape:", output.shape)  # 应该是 [1, 2, 256, 256]