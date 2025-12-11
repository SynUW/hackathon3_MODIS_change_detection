# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changemask.py
"""

import ever as er
import ever.module.loss as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import models

try:
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
except ImportError:
    print(f"segmentation_models_pytorch not found. please `pip install segmentation_models_pytorch`")



CHANGE = 'change_prediction'
T1SEM = 't1_semantic_prediction'
T2SEM = 't2_semantic_prediction'


def bitemporal_forward(module, x):
    x = rearrange(x, 'b (t c) h w -> (b t) c h w', t=2)
    features = module(x)
    if isinstance(features, list) or isinstance(features, tuple):
        t1_features, t2_features = [], []
        for feat in features:
            t1_feat, t2_feat = rearrange(feat, '(b t) c h w -> t b c h w', t=2)
            t1_features.append(t1_feat)
            t2_features.append(t2_feat)
        # 确保返回的是列表
        return t1_features, t2_features
    else:
        # 如果是单个特征图，将其转换为列表
        t1_features, t2_features = rearrange(features, '(b t) c h w -> t b c h w', t=2)
        return [t1_features], [t2_features]


@torch.cuda.amp.autocast(dtype=torch.float32)
def mse_loss(s1_logit, s2_logit, gt_masks):
    c_gt = gt_masks[-1].to(torch.float32).unsqueeze(1)

    s1_p = s1_logit.log_softmax(dim=1).exp()
    s2_p = s2_logit.log_softmax(dim=1).exp()

    diff = (s1_p - s2_p) ** 2
    losses = (1 - c_gt) * diff + c_gt * (1 - diff)

    return losses.mean()


@torch.cuda.amp.autocast(dtype=torch.float32)
def loss(
        s1_logit, s2_logit, c_logit,
        gt_masks,
):
    s1_gt = gt_masks[0].to(torch.int64)
    s2_gt = gt_masks[1].to(torch.int64)

    s1_ce = F.cross_entropy(s1_logit, s1_gt, ignore_index=255)
    s1_dice = L.dice_loss_with_logits(s1_logit, s1_gt)

    s2_ce = F.cross_entropy(s2_logit, s2_gt, ignore_index=255)
    s2_dice = L.dice_loss_with_logits(s2_logit, s2_gt)

    c_gt = gt_masks[-1].to(torch.float32)
    c_dice = L.dice_loss_with_logits(c_logit, c_gt)
    c_bce = L.binary_cross_entropy_with_logits(c_logit, c_gt)

    sim_loss = mse_loss(s1_logit, s2_logit, gt_masks)
    return {
        's1_ce_loss': s1_ce,
        's1_dice_loss': s1_dice,
        's2_ce_loss': s2_ce,
        's2_dice_loss': s2_dice,
        'c_dice_loss': c_dice,
        'c_bce_loss': c_bce,
        # to improve semantic-change consistency, this is a well-known issue in ChangeMask-like SCD methods.
        # original implementation doesn't have this objective.
        'sim_loss': sim_loss
    }


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.squeeze(dim=self.dim)


class SpatioTemporalInteraction(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 type='conv3d'):
        if type == 'conv3d':
            padding = dilation * (kernel_size - 1) // 2
            super(SpatioTemporalInteraction, self).__init__(
                nn.Conv3d(in_channels, out_channels, [2, kernel_size, kernel_size], stride=1,
                          dilation=(1, dilation, dilation),
                          padding=(0, padding, padding),
                          bias=False),
                Squeeze(dim=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        elif type == 'conv1plus2d':
            super(SpatioTemporalInteraction, self).__init__(
                nn.Conv3d(in_channels, out_channels, (2, 1, 1), stride=1,
                          padding=(0, 0, 0),
                          bias=False),
                Squeeze(dim=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, kernel_size, 1,
                          kernel_size // 2) if kernel_size > 1 else nn.Identity(),
                nn.BatchNorm2d(out_channels) if kernel_size > 1 else nn.Identity(),
                nn.ReLU(True) if kernel_size > 1 else nn.Identity(),
            )


class TemporalSymmetricTransformer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 interaction_type='conv3d',
                 symmetric_fusion='add'):
        super(TemporalSymmetricTransformer, self).__init__()

        if isinstance(in_channels, list) or isinstance(in_channels, tuple):
            self.t = nn.ModuleList([
                SpatioTemporalInteraction(inc, outc, kernel_size, dilation=dilation, type=interaction_type)
                for inc, outc in zip(in_channels, out_channels)
            ])
        else:
            self.t = SpatioTemporalInteraction(in_channels, out_channels, kernel_size, dilation=dilation,
                                               type=interaction_type)

        if symmetric_fusion == 'add':
            self.symmetric_fusion = lambda x, y: x + y
        elif symmetric_fusion == 'mul':
            self.symmetric_fusion = lambda x, y: x * y
        elif symmetric_fusion == None:
            self.symmetric_fusion = None

    def forward(self, features1, features2):
        if isinstance(features1, list):
            d12_features = [op(torch.stack([f1, f2], dim=2)) for op, f1, f2 in
                            zip(self.t, features1, features2)]
            if self.symmetric_fusion:
                d21_features = [op(torch.stack([f2, f1], dim=2)) for op, f1, f2 in
                                zip(self.t, features1, features2)]
                change_features = [self.symmetric_fusion(d12, d21) for d12, d21 in zip(d12_features, d21_features)]
            else:
                change_features = d12_features
        else:
            if self.symmetric_fusion:
                change_features = self.symmetric_fusion(self.t(torch.stack([features1, features2], dim=2)),
                                                        self.t(torch.stack([features2, features1], dim=2)))
            else:
                change_features = self.t(torch.stack([features1, features2], dim=2))
            change_features = change_features.squeeze(dim=2)
        return change_features


@er.registry.MODEL.register()
class ChangeMask(er.ERModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Load pretrained ResNet34
        resnet = models.resnet34(pretrained=True)
        
        # Modify first layer for different input channels
        if cfg['in_channels'] > 3:
            repeat_times = cfg['in_channels'] // 3
            remainder = cfg['in_channels'] % 3
            weight_data = resnet.conv1.weight.data.repeat(1, repeat_times, 1, 1)
            if remainder > 0:
                weight_data = torch.cat([weight_data, resnet.conv1.weight.data[:, :remainder, :, :]], dim=1)
            newconv1 = nn.Conv2d(cfg['in_channels'], 64, kernel_size=7, stride=2, padding=3, bias=False)
            newconv1.weight.data.copy_(weight_data)
        else:
            newconv1 = nn.Conv2d(cfg['in_channels'], 64, kernel_size=7, stride=2, padding=3, bias=False)
            newconv1.weight.data[:, :cfg['in_channels'], :, :].copy_(resnet.conv1.weight.data[:, :cfg['in_channels'], :, :])
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Sequential(newconv1, resnet.bn1, resnet.relu),
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Change detection head
        self.change_head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),  # Change input channels from 64 to 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1)  # Output 2 channels for change detection
        )
        
        # Classification heads
        self.classifier1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Change to 1x1 output
            nn.Conv2d(64, cfg['num_classes'], kernel_size=1)
        )
        
        self.classifier2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Change to 1x1 output
            nn.Conv2d(64, cfg['num_classes'], kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        # Encode both inputs
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        
        # Decode features
        x1 = self.decoder(x1)
        x2 = self.decoder(x2)
        
        # Concatenate features for change detection
        x = torch.cat([x1, x2], dim=1)
        
        # Get change detection output
        change = self.change_head(x)
        change = F.adaptive_avg_pool2d(change, (1, 1))  # Pool to 1x1
        
        # Get classification outputs
        out1 = self.classifier1(x1)
        out2 = self.classifier2(x2)

        return {
            'change': change.squeeze(),
            'class_2010': out1.squeeze(),
            'class_2015': out2.squeeze()
        }

    def set_default_config(self):
        self.cfg.update(dict(
            num_classes=11,
            in_channels=138
        ))


if __name__ == '__main__':
    # Test configuration
    config = {
        'num_classes': 11,
        'in_channels': 138
    }
    
    # Create model instance
    model = ChangeMask(config)  # Pass config as a single argument
    
    # Print model architecture
    print("Model Architecture:")
    print(model)
    
    # Test forward propagation
    batch_size = 4
    channels = 138
    height, width = 13, 13
    
    # Create test inputs
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    
    # Training mode test
    print("\nTraining Mode Test:")
    model.train()
    outputs = model(x1, x2)
    print(f"Change Detection Map shape: {outputs['change'].shape}")
    print(f"Phase 1 Semantic Segmentation shape: {outputs['class_2010'].shape}")
    print(f"Phase 2 Semantic Segmentation shape: {outputs['class_2015'].shape}")
    
    # Inference mode test
    print("\nInference Mode Test:")
    model.eval()
    with torch.no_grad():
        outputs = model(x1, x2)
    print(f"Change Detection Map shape: {outputs['change'].shape}")
    print(f"Phase 1 Semantic Segmentation shape: {outputs['class_2010'].shape}")
    print(f"Phase 2 Semantic Segmentation shape: {outputs['class_2015'].shape}")
    
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
        x1 = x1.cuda()
        x2 = x2.cuda()
        with torch.no_grad():
            outputs = model(x1, x2)
        print("GPU Test Successful!")
