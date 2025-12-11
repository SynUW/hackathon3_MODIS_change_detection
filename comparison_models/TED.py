"""
https://github.com/DingLei14/SCanNet/blob/main/models/TED.py
"""
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from .misc import initialize_weights

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels, scale_ratio=1):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low//scale_ratio
        self.transit = nn.Sequential(
            conv1x1(in_channels_low, in_channels_low//scale_ratio),
            nn.BatchNorm2d(in_channels_low//scale_ratio),
            nn.ReLU(inplace=True) )
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) )

    def forward(self, x, low_feat):
        x = self.up(x)
        low_feat = self.transit(low_feat)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)
        return x

class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained)
        
        # Modify first convolution layer
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Adjust pretrained weights
        if in_channels == 3:
            # For 3-channel input, adjust pretrained weights
            pretrained_weights = resnet.conv1.weight.data
            pretrained_weights = F.interpolate(pretrained_weights, size=(3, 3), mode='bilinear', align_corners=True)
            newconv1.weight.data.copy_(pretrained_weights)
        else:
            # For multi-channel input, initialize first 3 channels
            pretrained_weights = resnet.conv1.weight.data
            pretrained_weights = F.interpolate(pretrained_weights, size=(3, 3), mode='bilinear', align_corners=True)
            newconv1.weight.data[:, 0:3, :, :].copy_(pretrained_weights)
            # Initialize remaining channels with Kaiming initialization
            nn.init.kaiming_normal_(newconv1.weight.data[:, 3:, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Modify stride to 1 to reduce downsampling
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
                
        self.head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU()
        )
        initialize_weights(self.head)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class TED(nn.Module):
    def __init__(self, in_channels=138, num_classes=11):
        super(TED, self).__init__()
        self.FCN = FCN(in_channels, pretrained=True)
        
        # Classification heads
        self.classifier1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        self.classifier2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        
        # Change detection head
        self.resCD = self._make_layer(ResBlock, 256, 128, 3, stride=1)
        self.classifierCD = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 64, kernel_size=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.Conv2d(64, 2, kernel_size=1)  # 输出[B, 2, 1, 1]
        )
            
        initialize_weights(self.classifier1, self.classifier2, self.resCD, self.classifierCD)
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def base_forward(self, x):
        x = self.FCN.layer0(x)  # 13x13
        x = self.FCN.maxpool(x)  # 7x7
        x = self.FCN.layer1(x)  # 7x7
        x = self.FCN.layer2(x)  # 7x7
        x = self.FCN.layer3(x)  # 7x7
        x = self.FCN.layer4(x)  # 7x7
        x = self.FCN.head(x)    # 7x7
        return x
    
    def CD_forward(self, x1, x2):
        b,c,h,w = x1.size()
        x = torch.cat([x1,x2], 1)
        x = self.resCD(x)
        change = self.classifierCD(x)
        return change.squeeze(-1).squeeze(-1)  # 从[B, 2, 1, 1]变为[B, 2]
    
    def forward(self, x1, x2):
        x_size = x1.size()
        
        x1 = self.base_forward(x1)
        x2 = self.base_forward(x2)
        change = self.CD_forward(x1, x2)

        # Classification outputs
        out1 = self.classifier1(x1).squeeze(-1).squeeze(-1)  # [B, num_classes]
        out2 = self.classifier2(x2).squeeze(-1).squeeze(-1)  # [B, num_classes]
        
        return {
            'change': change,      # [B, 2]
            'class_2010': out1,    # [B, num_classes]
            'class_2015': out2     # [B, num_classes]
        }

if __name__ == '__main__':
    # Test configuration
    config = {
        'num_classes': 11,
        'in_channels': 138
    }
    
    # Create model instance
    model = TED(**config)
    
    # Print model architecture
    print("Model Architecture:")
    print(model)
    
    # Test forward pass
    batch_size = 2
    channels = 138
    height, width = 13, 13
    
    # Create test inputs
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    
    # Training mode test
    print("\nTraining Mode Test:")
    model.train()
    outputs = model(x1, x2)
    print(f"Change Detection Output Shape: {outputs['change'].shape}")
    print(f"Phase 1 Classification Shape: {outputs['class_2010'].shape}")
    print(f"Phase 2 Classification Shape: {outputs['class_2015'].shape}")
    
    # Inference mode test
    print("\nInference Mode Test:")
    model.eval()
    with torch.no_grad():
        outputs = model(x1, x2)
    print(f"Change Detection Output Shape: {outputs['change'].shape}")
    print(f"Phase 1 Classification Shape: {outputs['class_2010'].shape}")
    print(f"Phase 2 Classification Shape: {outputs['class_2015'].shape}")
    
    # Test model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters Statistics:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Test model on GPU if available
    if torch.cuda.is_available():
        print("\nGPU Test:")
        model = model.cuda()
        x1 = x1.cuda()
        x2 = x2.cuda()
        with torch.no_grad():
            outputs = model(x1, x2)
        print("GPU Test Successful!")
        
            
