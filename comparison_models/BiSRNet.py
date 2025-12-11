"""
https://github.com/DingLei14/Bi-SRNet
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

class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Handle different input channel sizes
        if in_channels > 3:
            # For channels > 3, repeat the RGB channels
            repeat_times = in_channels // 3
            remainder = in_channels % 3
            weight_data = resnet.conv1.weight.data.repeat(1, repeat_times, 1, 1)
            if remainder > 0:
                weight_data = torch.cat([weight_data, resnet.conv1.weight.data[:, :remainder, :, :]], dim=1)
            newconv1.weight.data.copy_(weight_data)
        else:
            # For channels <= 3, use original weights
            newconv1.weight.data[:, :in_channels, :, :].copy_(resnet.conv1.weight.data[:, :in_channels, :, :])
          
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

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x

class SR(nn.Module):
    '''Spatial reasoning module'''
    #codes from DANet 'Dual attention network for scene segmentation'
    def __init__(self, in_dim):
        super(SR, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        ''' inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = x+self.gamma*out        

        return out

class CotSR(nn.Module):
    #codes derived from DANet 'Dual attention network for scene segmentation'
    def __init__(self, in_dim):
        super(CotSR, self).__init__()
        self.chanel_in = in_dim

        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x1, x2):
        ''' inputs :
                x1 : input feature maps( B X C X H X W)
                x2 : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x1.size()
        
        q1 = self.query_conv1(x1).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        k1 = self.key_conv1(x1).view(m_batchsize, -1, width*height)
        v1 = self.value_conv1(x1).view(m_batchsize, -1, width*height)
        
        q2 = self.query_conv2(x2).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        k2 = self.key_conv2(x2).view(m_batchsize, -1, width*height)
        v2 = self.value_conv2(x2).view(m_batchsize, -1, width*height)
        
        energy1 = torch.bmm(q1, k2)
        attention1 = self.softmax(energy1)
        out1 = torch.bmm(v2, attention1.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)
                
        energy2 = torch.bmm(q2, k1)
        attention2 = self.softmax(energy2)
        out2 = torch.bmm(v1, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        
        out1 = x1 + self.gamma1*out1
        out2 = x2 + self.gamma2*out2  
        
        return out1, out2

class BiSRNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(BiSRNet, self).__init__()        
        self.FCN = FCN(in_channels, pretrained=True)
        self.SiamSR = SR(128)
        self.CotSR = CotSR(128)
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        
        self.res1 = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.classifier1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        self.classifier2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        
        self.CD = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.Conv2d(64, 2, kernel_size=1)
        )
        initialize_weights(self.head, self.SiamSR, self.res1, self.CD, self.CotSR, self.classifier1, self.classifier2)
    
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

    down_sample = nn.AdaptiveAvgPool2d((1, 1))
    
    def base_forward(self, x):
       
        x = self.FCN.layer0(x) #size:1/4
        x = self.FCN.maxpool(x) #size:1/4
        x = self.FCN.layer1(x) #size:1/4
        x = self.FCN.layer2(x) #size:1/8
        x = self.FCN.layer3(x) #size:1/16
        x = self.FCN.layer4(x)
        x = self.head(x)
        x = self.SiamSR(x)
        
        return x
    
    def CD_forward(self, x1, x2):
        b,c,h,w = x1.size()
        x = torch.cat([x1,x2], 1)
        x = self.res1(x)
        change = self.CD(x)
        return change
    
    def forward(self, x1, x2):
        x_size = x1.size()        
        x1 = self.base_forward(x1)
        x2 = self.base_forward(x2)
        change = self.CD_forward(x1, x2)
        change = self.down_sample(change)
        
        x1, x2 = self.CotSR(x1, x2)
        out1 = self.classifier1(x1)
        out2 = self.classifier2(x2)        
        
        return {
            'change': change.squeeze(),
            'class_2010': out1.squeeze(),
            'class_2015': out2.squeeze()
        }

if __name__ == '__main__':
    # Test configuration
    config = {
        'num_classes': 11,
        'in_channels': 138
    }
    
    # Create model instance
    model = BiSRNet(**config)
    
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