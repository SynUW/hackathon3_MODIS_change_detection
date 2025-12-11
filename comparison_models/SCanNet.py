"""
https://github.com/DingLei14/SCanNet/blob/main/models/SCanNet.py
需要512*512输入
"""
import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from .misc import initialize_weights
from .CSWin_Transformer import mit

args = {'hidden_size': 128*3,
        'mlp_dim': 256*3,
        'num_heads': 4,
        'num_layers': 2,
        'dropout_rate': 0.}
        
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
        resnet = models.resnet34(pretrained=pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 修改权重初始化方式
        if in_channels == 3:
            newconv1.weight.data.copy_(resnet.conv1.weight.data)
        else:
            # 对于多通道输入，使用前3个通道的权重，其余通道使用随机初始化
            newconv1.weight.data[:, :3, :, :].copy_(resnet.conv1.weight.data)
            nn.init.kaiming_normal_(newconv1.weight.data[:, 3:, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        initialize_weights(self.head)
                                  
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

class SCanNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, input_size=13):
        super(SCanNet, self).__init__()
        feat_size = 4  # Feature map size for transformer
        self.FCN = FCN(in_channels, pretrained=True)
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.transformer = mit(img_size=feat_size, in_chans=128*3, embed_dim=128*3)
        
        self.DecCD = _DecoderBlock(128, 128, 128, scale_ratio=2)
        self.Dec1  = _DecoderBlock(128, 64,  128)
        self.Dec2  = _DecoderBlock(128, 64,  128)
        
        # Classification heads
        self.classifierA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        self.classifierB = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        self.classifierCD = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1)
        )
            
        initialize_weights(self.Dec1, self.Dec2, self.classifierA, self.classifierB, self.resCD, self.DecCD, self.classifierCD)
    
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
        x = self.FCN.layer0(x)  # size: 1/2
        x = self.FCN.maxpool(x)  # size: 1/4
        x_low = self.FCN.layer1(x)  # size: 1/4
        x = self.FCN.layer2(x_low)  # size: 1/8
        x = self.FCN.layer3(x)
        x = self.FCN.layer4(x)
        x = self.FCN.head(x)
        return x, x_low
    
    def CD_forward(self, x1, x2):
        b,c,h,w = x1.size()
        x = torch.cat([x1,x2], 1)
        xc = self.resCD(x)
        return x1, x2, xc
    
    def forward(self, x1, x2):
        x_size = x1.size()
        
        x1, x1_low = self.base_forward(x1)
        x2, x2_low = self.base_forward(x2)
        x1, x2, xc = self.CD_forward(x1, x2)

        x1 = self.Dec1(x1, x1_low)
        x2 = self.Dec2(x2, x2_low)        
        xc_low = torch.cat([x1_low, x2_low], 1)
        xc = self.DecCD(xc, xc_low)
                
        x = torch.cat([x1, x2, xc], 1)
        
        # Adjust feature map size for transformer
        if x.size(2) != 4 or x.size(3) != 4:
            x = F.interpolate(x, size=(4, 4), mode='bilinear', align_corners=True)
        
        x = self.transformer(x)
        x1 = x[:, 0:128, :, :]
        x2 = x[:, 128:256, :, :]
        xc = x[:, 256:, :, :]
        
        # Get final outputs using global average pooling and 1x1 convolution
        out1 = self.classifierA(x1).squeeze(-1).squeeze(-1)  # [B, num_classes]
        out2 = self.classifierB(x2).squeeze(-1).squeeze(-1)  # [B, num_classes]
        change_pred = self.classifierCD(xc).squeeze(-1).squeeze(-1)  # [B, 2]

        return {
            'change': change_pred,  # B*2
            'class_2010': out1,  # B*C
            'class_2015': out2   # B*C
        }


if __name__ == '__main__':
    # Test configuration
    config = {
        'num_classes': 11,
        'in_channels': 138,
        'input_size': 13
    }
    
    # Create model instance
    model = SCanNet(**config)
    
    # Print model architecture
    print("Model Architecture:")
    print(model)
    
    # Test forward pass
    batch_size = 2
    channels = 138  # 138 channels input
    height, width = 13, 13  # 13x13 input
    
    # Create test inputs
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    
    # Training mode test
    print("\nTraining Mode Test:")
    model.train()
    output = model(x1, x2)  
    print(f"Change Detection Output Shape: {output['change'].shape}")
    print(f"Phase 1 Semantic Segmentation Shape: {output['class_2010'].shape}")
    print(f"Phase 2 Semantic Segmentation Shape: {output['class_2015'].shape}")
    
    # Inference mode test
    print("\nInference Mode Test:")
    model.eval()
    with torch.no_grad():
        change, out1, out2 = model(x1, x2)
    print(f"Change Detection Output Shape: {change.shape}")
    print(f"Phase 1 Semantic Segmentation Shape: {out1.shape}")
    print(f"Phase 2 Semantic Segmentation Shape: {out2.shape}")
    
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
            change, out1, out2 = model(x1, x2)
        print("GPU Test Successful!")
        