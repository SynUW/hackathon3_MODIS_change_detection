# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changesparse.py
"""

# Configure NumExpr threads
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'  # Set NumExpr threads to 16

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.amp import autocast
from timm.layers import DropPath
from timm.models.swin_transformer import window_partition, window_reverse, to_2tuple, WindowAttention

import ever as er
import ever.module as M
import ever.module.loss as L

from einops import rearrange
from segmentation_models_pytorch.encoders import get_encoder
import math
import numpy as np
from skimage import measure


class LossMixin:
    def loss(self, y_true: torch.Tensor, y_pred, loss_config):
        loss_dict = dict()

        if 'prefix' in loss_config:
            prefix = loss_config.prefix
        else:
            prefix = ''

        if 'mem' in loss_config:
            mem = torch.cuda.max_memory_allocated() // 1024 // 1024
            loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(y_pred.device)

        if 'bce' in loss_config:
            weight = loss_config.bce.get('weight', 1.0)
            if y_pred.size(1) == 2:
                # 对于二分类情况，确保y_true是B*2格式
                if y_true.dim() == 2:  # 如果是B*1格式
                    y_true = torch.stack([1 - y_true, y_true], dim=1)
                # 确保y_true和y_pred维度匹配
                if y_true.dim() == 3:  # 如果是B*2*2格式
                    y_true = y_true.mean(dim=-1)  # 转换为B*2
                loss_dict[f'{prefix}bce@w{weight}_loss'] = weight * F.binary_cross_entropy_with_logits(
                    y_pred,
                    y_true,
                    reduction='mean'
                )
            else:
                loss_dict[f'{prefix}bce@w{weight}_loss'] = weight * L.label_smoothing_binary_cross_entropy(
                    y_pred,
                    y_true.float(),
                    eps=loss_config.bce.get('label_smooth', 0.),
                    reduction='mean',
                    ignore_index=loss_config.ignore_index
                )
            del weight

        if 'ce' in loss_config:
            weight = loss_config.ce.get('weight', 1.0)
            # 确保y_true是Long类型，并且y_pred的shape是[B, C]
            if y_true.dim() == 1:
                y_true = y_true.long()
            else:
                y_true = y_true.argmax(dim=1).long()
            loss_dict[f'{prefix}ce@w{weight}_loss'] = weight * F.cross_entropy(y_pred, y_true, ignore_index=loss_config.ignore_index)
            del weight

        if 'dice' in loss_config:
            ignore_channel = loss_config.dice.get('ignore_channel', -1)
            weight = loss_config.dice.get('weight', 1.0)
            if y_pred.size(1) == 2:
                # 对于二分类情况，确保y_true是B*2格式
                if y_true.dim() == 2:  # 如果是B*1格式
                    y_true = torch.stack([1 - y_true, y_true], dim=1)
                # 确保y_true和y_pred维度匹配
                if y_true.dim() == 3:  # 如果是B*2*2格式
                    y_true = y_true.mean(dim=-1)  # 转换为B*2
                loss_dict[f'{prefix}dice@w{weight}_loss'] = weight * self._dice_loss_2channel(
                    y_pred, y_true,
                    ignore_index=loss_config.ignore_index,
                    ignore_channel=ignore_channel
                )
            else:
                loss_dict[f'{prefix}dice@w{weight}_loss'] = weight * L.dice_loss_with_logits(
                    y_pred, y_true.float(),
                    ignore_index=loss_config.ignore_index,
                    ignore_channel=ignore_channel)
            del weight

        if 'tver' in loss_config:
            alpha = loss_config.tver.alpha
            beta = round(1. - alpha, 2)
            weight = loss_config.tver.get('weight', 1.0)
            gamma = loss_config.tver.get('gamma', 1.0)
            smooth_value = loss_config.tver.get('smooth_value', 1.0)
            # 对于二分类情况，确保y_true是B*2格式
            if y_true.dim() == 2:  # 如果是B*1格式
                y_true = torch.stack([1 - y_true, y_true], dim=1)
            # 确保y_true和y_pred维度匹配
            if y_true.dim() == 3:  # 如果是B*2*2格式
                y_true = y_true.mean(dim=-1)  # 转换为B*2
            loss_dict[f'{prefix}tver[{alpha},{beta},{gamma}]@w{weight}_loss'] = weight * L.tversky_loss_with_logits(
                y_pred, y_true.float(),
                alpha, beta, gamma,
                smooth_value=smooth_value,
                ignore_index=loss_config.ignore_index,
            )
            del weight

        if 'log_binary_iou_sigmoid' in loss_config:
            with torch.no_grad():
                if y_pred.size(1) == 2:
                    _y_pred = y_pred
                    _y_true = y_true
                    _binary_y_true = _y_true.reshape(-1)
                    cls = (_y_pred.sigmoid() > 0.5).float().reshape(-1)
                else:
                    _y_pred, _y_true = L.select(y_pred, y_true, loss_config.ignore_index)
                    _binary_y_true = (_y_true > 0).float().reshape(-1)
                    cls = (_y_pred.sigmoid() > 0.5).float().reshape(-1)
            print('binary_y_true shape:', _binary_y_true.shape, 'cls shape:', cls.shape)
            loss_dict[f'{prefix}iou-1'] = self._iou_1(_binary_y_true, cls)
        return loss_dict

    @staticmethod
    def _dice_loss_2channel(y_pred, y_true, ignore_index=None, ignore_channel=-1):
        y_pred = y_pred.sigmoid()
        y_pred = y_pred.reshape(y_pred.size(0), y_pred.size(1), -1)
        y_true = y_true.reshape(y_true.size(0), y_true.size(1), -1)
        
        if ignore_index is not None:
            mask = (y_true != ignore_index).float()
            y_pred = y_pred * mask
            y_true = y_true * mask
        
        if ignore_channel >= 0:
            y_pred = y_pred[:, :ignore_channel]
            y_true = y_true[:, :ignore_channel]
        
        intersection = (y_pred * y_true).sum(dim=2)
        union = y_pred.sum(dim=2) + y_true.sum(dim=2)
        
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        return 1 - dice.mean()

    @staticmethod
    def _iou_1(y_true, y_pred, ignore_index=None):
        with torch.no_grad():
            if ignore_index:
                y_pred = y_pred.reshape(-1)
                y_true = y_true.reshape(-1)
                valid = y_true != ignore_index
                y_true = y_true.masked_select(valid).float()
                y_pred = y_pred.masked_select(valid).float()
            y_pred = y_pred.float().reshape(-1)
            y_true = y_true.float().reshape(-1)
            inter = torch.sum(y_pred * y_true)
            union = y_true.sum() + y_pred.sum()
            return inter / torch.max(union - inter, torch.as_tensor(1e-6, device=y_pred.device))


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SMPEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[2:]


def get_backbone(name, pretrained=False, **kwargs):
    if name == 'er.R50c':
        return M.ResNetEncoder(dict(
            resnet_type='resnet50_v1c',
            pretrained=pretrained,
            in_channels=kwargs.get('in_channels', 3)
        )), (256, 512, 1024, 2048)
    elif name == 'er.R18':
        return M.ResNetEncoder(dict(
            resnet_type='resnet18',
            pretrained=pretrained,
            in_channels=kwargs.get('in_channels', 3)
        )), (64, 128, 256, 512)
    elif name == 'er.R101c':
        return M.ResNetEncoder(dict(
            resnet_type='resnet101_v1c',
            pretrained=pretrained,
            in_channels=kwargs.get('in_channels', 3)
        )), (256, 512, 1024, 2048)
    elif name.startswith('efficientnet'):
        in_channels = kwargs.get('in_channels', 3)
        model = get_encoder(name=name, weights='imagenet' if pretrained else None, in_channels=in_channels)
        out_channels = model.out_channels[2:]
        model = SMPEncoder(model)
        return model, out_channels
    else:
        raise NotImplementedError(f'{name} is not supported now.')


class ADBN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(args[1])

    def forward(self, x):
        x = rearrange(x, 'b (t c) h w ->t b c h w', t=2)
        x = torch.abs(x[0] - x[1])
        x = self.bn(x)
        return x


class TemporalReduction(nn.Module):
    def __init__(self, single_temporal_in_channels, reduce_type='conv'):
        super().__init__()
        self.channels = single_temporal_in_channels
        if reduce_type == 'conv':
            op = M.ConvBlock
        elif reduce_type == 'ADBN':
            op = ADBN
        else:
            raise NotImplementedError

        self.temporal_convs = nn.ModuleList()
        for c in self.channels:
            self.temporal_convs.append(op(2 * c, c, 1, bias=False))

    def forward(self, features):
        return [tc(rearrange(f, '(b t) c h w -> b (t c) h w ', t=2)) for f, tc in zip(features, self.temporal_convs)]


class ConvMlp(Mlp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dwconv = nn.Conv2d(self.fc1.out_features,
                                self.fc1.out_features, 3, 1, 1, bias=False,
                                groups=self.fc1.out_features)

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = rearrange(x, 'b (h w) c ->b c h w', h=h, w=w).contiguous()
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w ->b (h w) c').contiguous()
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @autocast('cuda', dtype=torch.float32)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SparseAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def masked_attn(self, x, indices):
        device = x.device
        B, N, C = x.shape

        # 确保indices不会超出范围
        indices = torch.clamp(indices, 0, N-1)

        batch_range = torch.arange(B, device=device)[:, None]
        selected_x = x[batch_range, indices]
        selected_x = self.attn(selected_x)
        x[batch_range, indices] = selected_x

        return x

    def forward(self, x, indices):
        h, w = x.size(2), x.size(3)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x = x + self.drop_path(self.masked_attn(self.norm1(x), indices))
        x = self.norm2(x)
        x = x + self.drop_path(self.mlp(x, h, w))
        x = rearrange(x, 'b (h w) c ->b c h w', h=h, w=w).contiguous()
        return x


class DenseAttentionBlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop,
                         drop_path, act_layer, norm_layer)
        mlp_hidden_dim = int(dim * mlp_ratio)
        del self.mlp
        self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        x = rearrange(x, 'b (h w) c ->b c h w', h=h, w=w).contiguous()
        return x


class SwinAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = to_2tuple(window_size)  # 确保window_size是元组
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size[0], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # 处理输入形状
        if len(x.shape) == 4:  # 如果是4D张量 [B, C, H, W]
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        else:  # 如果是3D张量 [B, H*W, C]
            B, L, C = x.shape
            H = W = int(math.sqrt(L))
            assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 计算pad
        pad_l = pad_t = 0
        pad_r = (self.window_size[0] - W % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # 计算shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = None
        else:
            shifted_x = x
            attn_mask = None

        # 分割窗口
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # 还原shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # 如果输入是4D，则转换回4D
        if len(x.shape) == 3:
            x = x.transpose(1, 2).view(B, C, H, W)

        return x


class SimpleFusion(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_ratio=0.1):
        super(SimpleFusion, self).__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.dropout = nn.Dropout2d(p=dropout_ratio) if dropout_ratio > 0 else nn.Identity()

    def forward(self, feat_list):
        x0 = feat_list[0]
        x0_h, x0_w = x0.size(2), x0.size(3)
        feats = [x0]
        for feat in feat_list[1:]:
            xi = F.interpolate(feat, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
            feats.append(xi)

        x = torch.cat(feats, dim=1)
        x = self.fuse_conv(x)
        x = self.dropout(x)
        return x


class SparseChangeTransformer(nn.Module):
    def __init__(self,
                 in_channels_list,
                 inner_channels=192,
                 num_heads=(3, 3),
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 change_threshold=0.5,
                 min_keep_ratio=0.1,
                 max_keep_ratio=0.5,
                 train_max_keep=2000,
                 num_blocks=(2, 2),
                 disable_attn_refine=False,
                 output_type='single_scale',
                 pc_upsample='nearest',
                 ):
        super().__init__()
        self.pc_upsample = pc_upsample
        self.disable_attn_refine = disable_attn_refine
        self.train_max_keep = train_max_keep
        top_layers = [M.ConvBlock(in_channels_list[-1], inner_channels, 1, bias=False)]
        win_size = 8
        top_layers += [
            SwinAttentionBlock(inner_channels, num_heads[0],
                               window_size=win_size,
                               shift_size=0 if (i % 2 == 0) else win_size // 2,
                               mlp_ratio=4.,
                               qkv_bias=qkv_bias,
                               drop=drop,
                               attn_drop=attn_drop,
                               drop_path=drop_path)
            for i in range(num_blocks[0])
        ]
        self.top_attn = nn.Sequential(*top_layers)

        self.num_stages = len(in_channels_list) - 1
        self.region_predictor = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(inner_channels, 1, 1)
        )

        self.refine_stages = nn.ModuleList()

        for i in range(self.num_stages):
            stage = nn.ModuleList([
                SparseAttentionBlock(inner_channels, num_heads[min(i + 1, len(num_heads) - 1)], 4.0, qkv_bias, drop, attn_drop, drop_path)
                for _ in range(num_blocks[min(i + 1, len(num_blocks) - 1)])])
            self.refine_stages.append(stage)

        self.conv1x1s = nn.ModuleList(
            [M.ConvBlock(in_channels_list[i], inner_channels, 1, bias=False) for i in range(self.num_stages)])
        self.reduce_convs = nn.ModuleList(
            [M.ConvBlock(inner_channels * 2, inner_channels, 1, bias=False) for _ in range(self.num_stages)])

        self.change_threshold = change_threshold
        self.min_keep_ratio = min_keep_ratio
        self.max_keep_ratio = max_keep_ratio

        self.output_type = output_type
        if output_type == 'multi_scale':
            self.simple_fuse = SimpleFusion(inner_channels * 4, inner_channels)

        self.init_weight()

    def init_weight(self):
        prior_prob = 0.001
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.region_predictor[-1].bias, bias_value)

    def forward(self, features):
        outputs = [self.top_attn(features.pop(-1))]

        intermediate_logits = []
        estimated_change_ratios = []
        prob = None
        for i in range(len(features)):
            top = outputs[-(i + 1)]
            if i == 0:
                indices, logit, top_2x, ecr = self.change_region_predict(self.region_predictor, top)
                intermediate_logits.append(logit)
                estimated_change_ratios.append(ecr)
                prob = logit.softmax(dim=1)
            else:
                top_2x = F.interpolate(top, scale_factor=2., mode='nearest')
                prob = F.interpolate(prob, scale_factor=2., mode='nearest')
                indices, _ = self.prob2indices(prob)

            down = features.pop(-1)
            down = self.conv1x1s[-(i + 1)](down)
            
            # 确保top_2x和down的空间维度匹配
            if top_2x.size(2) != down.size(2) or top_2x.size(3) != down.size(3):
                top_2x = F.interpolate(top_2x, size=(down.size(2), down.size(3)), mode='nearest')
            
            down = self.reduce_convs[i](torch.cat([down, top_2x], dim=1))

            if not self.disable_attn_refine:
                # 确保indices的维度与特征图匹配
                h, w = down.size(2), down.size(3)
                if indices.size(1) > h * w:
                    indices = indices[:, :h*w]
                down = self.attention_refine(self.refine_stages[i], down, indices)

            outputs.insert(0, down)
            
        if self.output_type == 'single_scale':
            output = outputs[0]
        elif self.output_type == 'multi_scale':
            output = self.simple_fuse(outputs)
        else:
            raise ValueError()
            
        return {
            'output_feature': output,
            'intermediate_logits': intermediate_logits,
            'estimated_change_ratios': estimated_change_ratios
        }

    def change_region_predict(self, region_predictor, feature):
        feature = F.interpolate(feature, scale_factor=2., mode='nearest')

        change_region_logit = region_predictor(feature)
        change_region_prob = change_region_logit.softmax(dim=1)
        indices, estimated_change_ratio = self.prob2indices(change_region_prob)

        return indices, change_region_logit, feature, estimated_change_ratio

    def prob2indices(self, prob):
        h, w = prob.size(2), prob.size(3)
        total_pixels = h * w

        # 计算变化区域的数量
        max_num_change_regions = (prob > self.change_threshold).long().sum(dim=(1, 2, 3)).max().item()
        
        # 确保max_num_change_regions在合理范围内
        max_num_change_regions = max(int(self.min_keep_ratio * total_pixels),
                                   min(max_num_change_regions, int(self.max_keep_ratio * total_pixels)))
        
        # 确保不超过特征图大小
        max_num_change_regions = min(max_num_change_regions, total_pixels)

        # 在训练时限制最大数量
        if self.training:
            max_num_change_regions = min(self.train_max_keep, max_num_change_regions)

        # 计算估计的变化比例
        estimated_change_ratio = max_num_change_regions / total_pixels

        # 获取top-k索引，并确保索引在有效范围内
        prob_flat = prob.flatten(2)  # B x C x (H*W)
        _, indices = torch.topk(prob_flat[:, 0], k=max_num_change_regions, dim=1, largest=True)
        
        return indices, estimated_change_ratio

    def attention_refine(self, refine_blocks, feature, indices):
        for op in refine_blocks:
            feature = op(feature, indices)
        return feature


@er.registry.MODEL.register()
class ChangeSparseBCD(er.ERModule, LossMixin):
    def __init__(self, config):
        super().__init__(config)

        self.backbone, channels = get_backbone(
            self.cfg.backbone.name,
            self.cfg.backbone.pretrained,
            in_channels=self.cfg.backbone.get('in_channels', 3)
        )
        self.temporal_reduce = TemporalReduction(channels, self.cfg.temporal_reduction.reduce_type)
        self.multi_stage_attn = SparseChangeTransformer(
            channels,
            **self.cfg.transformer,
        )
        self.conv_change = M.ConvUpsampling(self.cfg.transformer.inner_channels, 1, 4, 1)

    def forward(self, x, y=None):
        x = rearrange(x, 'b (t c) h w -> (b t) c h w', t=2)
        x = self.backbone(x)
        x = self.temporal_reduce(x)
        outputs = self.multi_stage_attn(x)

        output_feature = outputs['output_feature']

        logit = self.conv_change(output_feature)

        if self.training:
            gt_change = (y['masks'][-1] > 0).float()

            loss_dict = self.loss(gt_change, logit, self.config.main_loss)

            # region loss
            for i, region_logit in enumerate(outputs['intermediate_logits']):
                h, w = region_logit.size(2), region_logit.size(3)
                gt_region_change = F.adaptive_max_pool2d(gt_change.unsqueeze(0), (h, w)).squeeze(0)
                self.config.region_loss[i].prefix = f'{h}x{w}_'
                loss_dict.update(self.loss(gt_region_change, region_logit, self.config.region_loss[i]))

            # log estimated change ratio
            for region_logit, ecr in zip(outputs['intermediate_logits'], outputs['estimated_change_ratios']):
                h, w = region_logit.size(2), region_logit.size(3)
                loss_dict.update({
                    f'{h}x{w}_ECR': torch.as_tensor(ecr).to(region_logit.device)
                })
            return loss_dict

        return {
            'change_prediction': logit.sigmoid()
        }

    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                name='er.R18',
                pretrained=True,
                drop_path_rate=0.,
                in_channels=138,  # 修改为138个通道
                stride=8  # 减少下采样次数，原来是32
            ),
            temporal_reduction=dict(
                reduce_type='conv'
            ),
            transformer=dict(
                inner_channels=96,
                num_heads=(3, 3),  # 减少注意力头数
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                change_threshold=0.5,
                min_keep_ratio=0.01,
                max_keep_ratio=0.1,
                train_max_keep=2000,
                num_blocks=(2, 2),  # 减少transformer层数
                disable_attn_refine=False,
                output_type='single_scale'
            ),
            main_loss=dict(
                bce=dict(),
                dice=dict(),
                mem=dict(),
                log_binary_iou_sigmoid=dict(),
                ignore_index=-1
            ),
            region_loss=[
                dict(
                    ignore_index=-1,
                    prefix='1'
                )
            ]
        ))

    def log_info(self):
        return {
            'encoder': self.backbone,
            'decoder': self.multi_stage_attn
        }

    def custom_param_groups(self):
        if self.cfg.backbone.name.startswith('mit'):
            param_groups = [{'params': [], 'weight_decay': 0.}, {'params': []}]
            for n, p in self.named_parameters():
                if 'norm' in n:
                    param_groups[0]['params'].append(p)
                elif 'pos_block' in n:
                    param_groups[0]['params'].append(p)
                else:
                    param_groups[1]['params'].append(p)
            return param_groups
        elif self.cfg.backbone.name.startswith('swin'):
            param_groups = [{'params': [], 'weight_decay': 0.}, {'params': []}]
            for i, p in self.named_parameters():
                if 'norm' in i:
                    param_groups[0]['params'].append(p)
                elif 'relative_position_bias_table' in i:
                    param_groups[0]['params'].append(p)
                elif 'absolute_pos_embed' in i:
                    param_groups[0]['params'].append(p)
                else:
                    param_groups[1]['params'].append(p)
            return param_groups
        else:
            return self.parameters()


class ChangeSparseTransformer_multiclass_impl(nn.Module):
    def __init__(
            self,
            in_channels_list,
            inner_channels=192,
            num_heads=(3, 3, 3, 3),
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            change_threshold=0.5,
            min_keep_ratio=0.1,
            max_keep_ratio=0.5,
            train_max_keep=2000,
            num_blocks=(2, 2, 2, 2),
            disable_attn_refine=False,
            output_type='single_scale'
    ):
        super().__init__()
        self.disable_attn_refine = disable_attn_refine
        self.train_max_keep = train_max_keep
        top_layers = [M.ConvBlock(in_channels_list[-1], inner_channels, 1, bias=False)]
        win_size = 8
        top_layers += [
            SwinAttentionBlock(inner_channels, num_heads[0],
                               window_size=win_size,
                               shift_size=0 if (i % 2 == 0) else win_size // 2,
                               mlp_ratio=4.,
                               qkv_bias=qkv_bias,
                               drop=drop,
                               attn_drop=attn_drop,
                               drop_path=drop_path)
            for i in range(num_blocks[0])
        ]
        self.top_attn = nn.Sequential(*top_layers)

        self.num_stages = len(in_channels_list) - 1
        self.region_predictor = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(inner_channels, 5, 1)
        )

        self.refine_stages = nn.ModuleList()

        for i in range(self.num_stages):
            stage = nn.ModuleList([
                SparseAttentionBlock(inner_channels, num_heads[i + 1], 4.0, qkv_bias, drop, attn_drop, drop_path)
                for _ in range(num_blocks[i + 1])])
            self.refine_stages.append(stage)

        self.conv1x1s = nn.ModuleList(
            [M.ConvBlock(in_channels_list[i], inner_channels, 1, bias=False) for i in range(self.num_stages)])
        self.reduce_convs = nn.ModuleList(
            [M.ConvBlock(inner_channels * 2, inner_channels, 1, bias=False) for _ in range(self.num_stages)])

        self.change_threshold = change_threshold
        self.min_keep_ratio = min_keep_ratio
        self.max_keep_ratio = max_keep_ratio

        self.output_type = output_type
        if output_type == 'multi_scale':
            self.simple_fuse = SimpleFusion(inner_channels * 4, inner_channels)

        self.init_weight()

    def init_weight(self):
        prior_prob = 0.001
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.region_predictor[-1].bias, bias_value)

    def forward(self, features):
        outputs = [self.top_attn(features.pop(-1))]

        intermediate_logits = []
        estimated_change_ratios = []
        prob = None
        for i in range(len(features)):
            top = outputs[-(i + 1)]
            if i == 0:
                indices, logit, top_2x, ecr = self.change_region_predict(self.region_predictor, top)
                intermediate_logits.append(logit)
                estimated_change_ratios.append(ecr)
                prob = logit.softmax(dim=1)
            else:
                top_2x = F.interpolate(top, scale_factor=2., mode='nearest')
                prob = F.interpolate(prob, scale_factor=2., mode='nearest')

                indices, _ = self.multi_class_prob2indices(prob)

            down = features.pop(-1)
            down = self.conv1x1s[-(i + 1)](down)
            down = self.reduce_convs[i](torch.cat([down, top_2x], dim=1))

            if not self.disable_attn_refine:
                down = self.attention_refine(self.refine_stages[i], down, indices)

            outputs.insert(0, down)
        if self.output_type == 'single_scale':
            output = outputs[0]
        elif self.output_type == 'multi_scale':
            output = self.simple_fuse(outputs)
        else:
            raise ValueError()
        return {
            'output_feature': output,
            'intermediate_logits': intermediate_logits,
            'estimated_change_ratios': estimated_change_ratios
        }

    def change_region_predict(self, region_predictor, feature):
        feature = F.interpolate(feature, scale_factor=2., mode='nearest')

        change_region_logit = region_predictor(feature)
        change_region_prob = change_region_logit.softmax(dim=1)
        indices, estimated_change_ratio = self.multi_class_prob2indices(change_region_prob)

        return indices, change_region_logit, feature, estimated_change_ratio

    def prob2indices(self, prob):
        h, w = prob.size(2), prob.size(3)
        total_pixels = h * w

        # 计算变化区域的数量
        max_num_change_regions = (prob > self.change_threshold).long().sum(dim=(1, 2, 3)).max().item()
        
        # 确保max_num_change_regions在合理范围内
        max_num_change_regions = max(int(self.min_keep_ratio * total_pixels),
                                   min(max_num_change_regions, int(self.max_keep_ratio * total_pixels)))
        
        # 确保不超过特征图大小
        max_num_change_regions = min(max_num_change_regions, total_pixels)

        # 在训练时限制最大数量
        if self.training:
            max_num_change_regions = min(self.train_max_keep, max_num_change_regions)

        # 计算估计的变化比例
        estimated_change_ratio = max_num_change_regions / total_pixels

        # 获取top-k索引，并确保索引在有效范围内
        prob_flat = prob.flatten(2)  # B x C x (H*W)
        _, indices = torch.topk(prob_flat[:, 0], k=max_num_change_regions, dim=1, largest=True)
        
        return indices, estimated_change_ratio

    def multi_class_prob2indices(self, prob):
        h, w = prob.size(2), prob.size(3)

        # max_num_change_regions = (prob.argmax(dim=1) > 1).long().sum(dim=(1, 2)).max().item()
        max_num_change_regions = (prob.argmax(dim=1) > 0).long().sum(dim=(1, 2)).max().item()

        max_num_change_regions = max(int(self.min_keep_ratio * h * w),
                                     min(max_num_change_regions, int(self.max_keep_ratio * h * w)))

        estimated_change_ratio = max_num_change_regions / (h * w)

        if self.training:
            max_num_change_regions = min(self.train_max_keep, max_num_change_regions)
        max_prob, _ = torch.max(prob, dim=1, keepdim=True)
        indices = torch.argsort(max_prob.flatten(2), dim=-1, descending=True)[:, 0, :max_num_change_regions]
        return indices, estimated_change_ratio

    def attention_refine(self, refine_blocks, feature, indices):
        for op in refine_blocks:
            feature = op(feature, indices)
        return feature


class SemanticDecoder(nn.Module):
    def __init__(self,
                 in_channels_list,
                 inner_channels=192,
                 num_heads=(3,),
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.1,
                 num_blocks=(2,),
                 output_type='single_scale'
                 ):
        super().__init__()
        top_layers = [M.ConvBlock(in_channels_list[-1], inner_channels, 1, bias=False)]
        win_size = 8
        top_layers += [
            SwinAttentionBlock(inner_channels, num_heads[0],
                               window_size=win_size,
                               shift_size=0 if (i % 2 == 0) else win_size // 2,
                               mlp_ratio=4.,
                               qkv_bias=qkv_bias,
                               drop=drop,
                               attn_drop=attn_drop,
                               drop_path=drop_path)
            for i in range(num_blocks[0])
        ]
        self.top_attn = nn.Sequential(*top_layers)

        self.num_stages = len(in_channels_list) - 1

        self.conv1x1s = nn.ModuleList(
            [M.ConvBlock(in_channels_list[i], inner_channels, 1, bias=False) for i in range(self.num_stages)])
        self.reduce_convs = nn.ModuleList(
            [M.ConvBlock(inner_channels * 2, inner_channels, 1, bias=False) for _ in range(self.num_stages)])

        self.output_type = output_type
        if output_type == 'multi_scale':
            self.simple_fuse = SimpleFusion(inner_channels * 4, inner_channels)

    def forward(self, features):
        j = -1
        outputs = [self.top_attn(features[j])]

        for i in range(len(features) - 1):
            top = outputs[-(i + 1)]
            top_2x = F.interpolate(top, scale_factor=2., mode='nearest')
            j -= 1
            down = features[j]
            down = self.conv1x1s[-(i + 1)](down)
            
            # 确保top_2x和down的空间维度匹配
            if top_2x.size(2) != down.size(2) or top_2x.size(3) != down.size(3):
                top_2x = F.interpolate(top_2x, size=(down.size(2), down.size(3)), mode='nearest')
            
            down = self.reduce_convs[i](torch.cat([down, top_2x], dim=1))
            outputs.insert(0, down)
            
        if self.output_type == 'single_scale':
            output = outputs[0]
        elif self.output_type == 'multi_scale':
            output = self.simple_fuse(outputs)
        else:
            raise ValueError()
        return {
            'output_feature': output,
        }


def object_based_infer(pre_logit, post_logit, logit_input=True):
    loc_thresh = 0. if logit_input else 0.5
    loc = (pre_logit > loc_thresh).cpu().squeeze(1).numpy()
    dam = post_logit.argmax(dim=1).cpu().squeeze(1).numpy()

    refined_dam = np.zeros_like(dam)
    for i, (single_loc, single_dam) in enumerate(zip(loc, dam)):
        refined_dam[i, :, :] = _object_vote(single_loc, single_dam)

    return loc, refined_dam


def _object_vote(loc, dam):
    damage_cls_list = [1, 2, 3, 4]
    local_mask = loc
    labeled_local, nums = measure.label(local_mask, connectivity=2, background=0, return_num=True)
    region_idlist = np.unique(labeled_local)
    if len(region_idlist) > 1:
        dam_mask = dam
        new_dam = local_mask.copy()
        for region_id in region_idlist:
            if all(local_mask[local_mask == region_id]) == 0:
                continue
            region_dam_count = [int(np.sum(dam_mask[labeled_local == region_id] == dam_cls_i)) * cls_weight \
                                for dam_cls_i, cls_weight in zip(damage_cls_list, [8., 38., 25., 11.])]
            dam_index = np.argmax(region_dam_count) + 1
            new_dam = np.where(labeled_local == region_id, dam_index, new_dam)
    else:
        new_dam = local_mask.copy()
    return new_dam


class FuseConv(nn.Sequential):
    def __init__(self, inchannels, outchannels):
        super(FuseConv, self).__init__(nn.Conv2d(inchannels, outchannels, kernel_size=1),
                                       nn.BatchNorm2d(outchannels),
                                       )
        self.relu = nn.ReLU(True)
        self.se = M.SEBlock(outchannels, 16)

    def forward(self, x):
        out = super(FuseConv, self).forward(x)
        residual = out
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


@er.registry.MODEL.register()
class ChangeSparseO2M(er.ERModule, LossMixin):
    def __init__(self, config):
        super().__init__(config)

        self.backbone, channels = get_backbone(
            self.cfg.backbone.name,
            self.cfg.backbone.pretrained,
            in_channels=self.cfg.backbone.get('in_channels', 3)
        )

        self.temporal_reduce = TemporalReduction(channels, self.cfg.temporal_reduction.reduce_type)
        self.multi_stage_attn = ChangeSparseTransformer_multiclass_impl(
            channels,
            **self.cfg.transformer,
        )
        self.conv_change = M.ConvUpsampling(self.cfg.transformer.inner_channels, 1, 4, 1)

        self.semantic_decoder = SemanticDecoder(
            channels,
            **self.cfg.semantic_decoder.transformer
        )
        c = self.cfg.semantic_decoder.transformer.inner_channels
        self.conv_loc = M.ConvUpsampling(c, 1, 4, 1)

        c1 = self.cfg.semantic_decoder.transformer.inner_channels
        c2 = self.cfg.transformer.inner_channels

        self.fuse_conv = FuseConv(c1 + c2, c2)

    def forward(self, x, y=None):
        x = rearrange(x, 'b (t c) h w -> (b t) c h w', t=2)
        x = self.backbone(x)
        semantic_features = self.semantic_decoder(x)

        x = self.temporal_reduce(x)

        outputs = self.multi_stage_attn(x)
        change_feature = outputs['output_feature']

        semantic_feature = semantic_features['output_feature']

        semantic_feature = rearrange(semantic_feature, '(b t) c h w -> t b c h w', t=2)

        semantic_feature1 = torch.cat([semantic_feature[0], change_feature], dim=1)
        semantic_feature2 = torch.cat([semantic_feature[1], change_feature], dim=1)

        semantic_logit1 = self.conv_semantic(semantic_feature1)
        semantic_logit2 = self.conv_semantic(semantic_feature2)

        logit = self.conv_change(change_feature)

        # 推理时也进行全局平均池化
        change_pred = F.adaptive_avg_pool2d(logit, (1, 1)).squeeze(-1).squeeze(-1)  # B*2
        t1_pred = F.adaptive_avg_pool2d(semantic_logit1, (1, 1)).squeeze(-1).squeeze(-1)  # B*C
        t2_pred = F.adaptive_avg_pool2d(semantic_logit2, (1, 1)).squeeze(-1).squeeze(-1)  # B*C

        return {
            'change': change_pred,  # B*2
            'class_2010': t1_pred,  # B*C
            'class_2015': t2_pred   # B*C
        }

    def postprocess(self, prob, changeos_mode=False):
        pre_prob = prob[:, 0:1, :, :]
        post_prob = prob[:, 1:, :, :]
        if changeos_mode:
            pr_loc, pr_dam = object_based_infer(pre_prob, post_prob, logit_input=False)
            return torch.from_numpy(pr_loc), torch.from_numpy(pr_dam)

        pr_loc = pre_prob > 0.5
        pr_dam = post_prob.argmax(dim=1, keepdim=True)
        return pr_loc, pr_dam

    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                name='er.R18',
                pretrained=True,
                drop_path_rate=0.,
                in_channels=138,  # 修改为138个通道
                stride=8  # 减少下采样次数，原来是32
            ),
            temporal_reduction=dict(
                reduce_type='conv'
            ),
            transformer=dict(
                inner_channels=96,
                num_heads=(3, 3),  # 减少注意力头数
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                change_threshold=0.5,
                min_keep_ratio=0.01,
                max_keep_ratio=0.1,
                train_max_keep=2000,
                num_blocks=(2, 2),  # 减少transformer层数
                disable_attn_refine=False,
                output_type='single_scale'
            ),
            num_change_classes=5,
            return_probs=False,
            changeos_mode=False,
            semantic_decoder=dict(
                transformer=dict(
                    inner_channels=96 * 2,
                    num_heads=(3,),
                    qkv_bias=True,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=0.1,
                    num_blocks=(2,),
                    output_type='single_scale'
                ),
            ),
            main_loss=dict(
            ),
            t1_seg_loss=dict(
                prefix='t1_',
                ce=dict(),
                ignore_index=-1
            ),
            t2_seg_loss=dict(
                prefix='t2_',
                ce=dict(),
                ignore_index=-1
            ),
            region_loss=[
                dict(
                    ignore_index=-1,
                    prefix='1'
                )
            ]
        ))

    def log_info(self):
        return {
            'encoder': self.backbone,
            'change_decoder': self.multi_stage_attn,
            'semantic_decoder': self.semantic_decoder
        }

    def custom_param_groups(self):
        if self.cfg.backbone.name.startswith('mit'):
            param_groups = [{'params': [], 'weight_decay': 0.}, {'params': []}]
            for n, p in self.named_parameters():
                if 'norm' in n:
                    param_groups[0]['params'].append(p)
                elif 'pos_block' in n:
                    param_groups[0]['params'].append(p)
                else:
                    param_groups[1]['params'].append(p)
            return param_groups
        else:
            return self.parameters()


@er.registry.MODEL.register()
class ChangeSparseM2M(er.ERModule, LossMixin):
    def __init__(self, config):
        super().__init__(config)

        self.backbone, channels = get_backbone(
            self.cfg.backbone.name,
            self.cfg.backbone.pretrained,
            in_channels=self.cfg.backbone.get('in_channels', 3)
        )

        self.temporal_reduce = TemporalReduction(channels, self.cfg.temporal_reduction.reduce_type)
        self.multi_stage_attn = SparseChangeTransformer(
            channels,
            **self.cfg.transformer,
        )

        self.semantic_decoder = SemanticDecoder(channels,
                                                **self.cfg.semantic_decoder.transformer
                                                )

        c = self.cfg.semantic_decoder.transformer.inner_channels
        self.conv_semantic = M.ConvUpsampling(c + self.cfg.transformer.inner_channels,
                                              self.cfg.semantic_decoder.num_classes, 4, 1)
        self.conv_change = M.ConvUpsampling(self.cfg.transformer.inner_channels, 2, 4, 1)

    def forward(self, x, y=None):
        x = rearrange(x, 'b (t c) h w -> (b t) c h w', t=2)
        x = self.backbone(x)
        semantic_features = self.semantic_decoder(x)

        x = self.temporal_reduce(x)

        outputs = self.multi_stage_attn(x)
        change_feature = outputs['output_feature']

        semantic_feature = semantic_features['output_feature']

        semantic_feature = rearrange(semantic_feature, '(b t) c h w -> t b c h w', t=2)

        semantic_feature1 = torch.cat([semantic_feature[0], change_feature], dim=1)
        semantic_feature2 = torch.cat([semantic_feature[1], change_feature], dim=1)

        semantic_logit1 = self.conv_semantic(semantic_feature1)
        semantic_logit2 = self.conv_semantic(semantic_feature2)

        logit = self.conv_change(change_feature)

        # 推理时也进行全局平均池化
        change_pred = F.adaptive_avg_pool2d(logit, (1, 1)).squeeze(-1).squeeze(-1)  # B*2
        t1_pred = F.adaptive_avg_pool2d(semantic_logit1, (1, 1)).squeeze(-1).squeeze(-1)  # B*C
        t2_pred = F.adaptive_avg_pool2d(semantic_logit2, (1, 1)).squeeze(-1).squeeze(-1)  # B*C

        return {
            'change': change_pred,  # B*2
            'class_2010': t1_pred,  # B*C
            'class_2015': t2_pred   # B*C
        }

    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                name='er.R18',
                pretrained=True,
                drop_path_rate=0.,
                in_channels=138,  # 修改为138个通道
                stride=8  # 减少下采样次数，原来是32
            ),
            temporal_reduction=dict(
                reduce_type='conv'
            ),
            transformer=dict(
                inner_channels=96,
                num_heads=(3, 3),  # 减少注意力头数
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                change_threshold=0.5,
                min_keep_ratio=0.01,
                max_keep_ratio=0.1,
                train_max_keep=2000,
                num_blocks=(2, 2),  # 减少transformer层数
                disable_attn_refine=False,
                output_type='single_scale'
            ),
            semantic_decoder=dict(
                num_classes=6,
                transformer=dict(
                    inner_channels=96 * 2,
                    num_heads=(3,),
                    qkv_bias=True,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=0.1,
                    num_blocks=(2,),
                    output_type='single_scale'
                ),
            ),
            main_loss=dict(
                bce=dict(),
                dice=dict(),
                mem=dict(),
                log_binary_iou_sigmoid=dict(),
                ignore_index=-1
            ),
            t1_seg_loss=dict(
                prefix='t1_',
                ce=dict(),
                ignore_index=-1
            ),
            t2_seg_loss=dict(
                prefix='t2_',
                ce=dict(),
                ignore_index=-1
            ),
            region_loss=[
                dict(
                    ignore_index=-1,
                    prefix='1'
                )
            ]
        ))

    def log_info(self):
        return {
            'encoder': self.backbone,
            'change_decoder': self.multi_stage_attn,
            'semantic_decoder': self.semantic_decoder
        }

    def custom_param_groups(self):
        if self.cfg.backbone.name.startswith('mit'):
            param_groups = [{'params': [], 'weight_decay': 0.}, {'params': []}]
            for n, p in self.named_parameters():
                if 'norm' in n:
                    param_groups[0]['params'].append(p)
                elif 'pos_block' in n:
                    param_groups[0]['params'].append(p)
                else:
                    param_groups[1]['params'].append(p)
            return param_groups
        else:
            return self.parameters()

if __name__ == '__main__':
    # Test configuration
    config = {
        'backbone': {
            'name': 'er.R18',
            'pretrained': False,
            'drop_path_rate': 0.,
            'in_channels': 138,  # 修改为138个通道
            'stride': 8  # 减少下采样次数，原来是32
        },
        'temporal_reduction': {
            'reduce_type': 'conv'
        },
        'transformer': {
            'inner_channels': 96,
            'num_heads': (3, 3),  # 减少注意力头数
            'qkv_bias': True,
            'drop': 0.,
            'attn_drop': 0.,
            'drop_path': 0.,
            'change_threshold': 0.5,
            'min_keep_ratio': 0.01,
            'max_keep_ratio': 0.1,
            'train_max_keep': 2000,
            'num_blocks': (2, 2),  # 减少transformer层数
            'disable_attn_refine': False,
            'output_type': 'single_scale'
        },
        'semantic_decoder': {
            'num_classes': 11,
            'transformer': {
                'inner_channels': 96 * 2,
                'num_heads': (3,),
                'qkv_bias': True,
                'drop': 0.,
                'attn_drop': 0.,
                'drop_path': 0.1,
                'num_blocks': (2,),
                'output_type': 'single_scale'
            }
        },
        'main_loss': {
            'bce': {},
            'dice': {},
            'mem': {},
            'log_binary_iou_sigmoid': {},
            'ignore_index': -1
        },
        't1_seg_loss': {
            'prefix': 't1_',
            'ce': {},
            'ignore_index': -1
        },
        't2_seg_loss': {
            'prefix': 't2_',
            'ce': {},
            'ignore_index': -1
        },
        'region_loss': [
            dict(
                ignore_index=-1,
                prefix='1'
            )
        ]
    }
    
    # Create model instance
    model = ChangeSparseM2M(config)
    
    # Print model architecture
    # print("Model Architecture:")
    # print(model)
    
    # Test forward propagation
    batch_size = 4
    channels = 138  # Input channels
    height, width = 13, 13
    
    # Create test inputs
    x = torch.randn(batch_size, channels * 2, height, width)  # Two temporal inputs concatenated
    y = {
        'masks': [
            torch.randint(0, 11, (batch_size, 11)),  # t1 semantic labels (B*C)
            torch.randint(0, 11, (batch_size, 11)),  # t2 semantic labels (B*C)
            torch.randint(0, 2, (batch_size, 2)).float()  # change mask (B*2)
        ]
    }
    
    # Training mode test
    print("\nTraining Mode Test:")
    model.train()
    loss_dict = model(x, y)
    print("Training Losses:")
    for k, v in loss_dict.items():
        print(f"{k}: {v.item():.4f}")
    
    # Inference mode test
    print("\nInference Mode Test:")
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    print("Output Keys:", outputs.keys())
    print(f"Change Detection Map shape: {outputs['change'].shape}")  # Should be B*2
    print(f"T1 Semantic Prediction shape: {outputs['class_2010'].shape}")  # Should be B*C
    print(f"T2 Semantic Prediction shape: {outputs['class_2015'].shape}")  # Should be B*C
    
    # Test model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters Statistics:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Test model on different devices
    if torch.cuda.is_available():
        print("\nGPU Test:")
        model = model.cuda()
        x = x.cuda()
        y = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in y.items()}
        with torch.no_grad():
            outputs = model(x)
        print("GPU Test Successful!")