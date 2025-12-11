"""
https://www.sciencedirect.com/science/article/pii/S1077314219300992
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions and skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class FC_EF_Res(nn.Module):
    """Fully Convolutional Early Fusion network with Residual blocks"""
    def __init__(self, input_channels=6, num_classes=1):
        super(FC_EF_Res, self).__init__()
        
        # Encoder
        self.encoder1 = nn.Sequential(
            ResidualBlock(input_channels, 32),
            nn.MaxPool2d(2, 2)
        )
        self.encoder2 = nn.Sequential(
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, 2)
        )
        self.encoder3 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, 2)
        )
        self.encoder4 = nn.Sequential(
            ResidualBlock(128, 256),
            nn.MaxPool2d(2, 2)
        )
        
        # Bottleneck
        self.bottleneck = ResidualBlock(256, 512)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = ResidualBlock(512, 256)  # 512 = 256(upconv) + 256(skip)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock(256, 128)  # 256 = 128(upconv) + 128(skip)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(128, 64)   # 128 = 64(upconv) + 64(skip)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(64, 32)    # 64 = 32(upconv) + 32(skip)
        
        # Final prediction
        self.conv_out = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        # 确保特征尺寸匹配
        if dec4.size() != enc4.size():
            dec4 = F.interpolate(dec4, size=enc4.size()[2:], mode='bilinear', align_corners=True)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        if dec3.size() != enc3.size():
            dec3 = F.interpolate(dec3, size=enc3.size()[2:], mode='bilinear', align_corners=True)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        if dec2.size() != enc2.size():
            dec2 = F.interpolate(dec2, size=enc2.size()[2:], mode='bilinear', align_corners=True)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        if dec1.size() != enc1.size():
            dec1 = F.interpolate(dec1, size=enc1.size()[2:], mode='bilinear', align_corners=True)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # Output
        out = self.conv_out(dec1)
        return out

class Integrated_LCM_CD(nn.Module):
    """Integrated Land Cover Mapping and Change Detection Network"""
    def __init__(self, input_channels=3, num_lcm_classes=6, num_cd_classes=1):
        super(Integrated_LCM_CD, self).__init__()
        
        # Shared encoder for both images
        self.encoder1 = nn.Sequential(
            ResidualBlock(input_channels, 32),
            nn.MaxPool2d(2, 2)
        )
        self.encoder2 = nn.Sequential(
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, 2)
        )
        self.encoder3 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, 2)
        )
        self.encoder4 = nn.Sequential(
            ResidualBlock(128, 256),
            nn.MaxPool2d(2, 2)
        )
        
        # Bottleneck for LCM (shared weights for both images)
        self.lcm_bottleneck = ResidualBlock(256, 512)
        
        # Decoder for LCM (shared weights for both images)
        self.lcm_upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.lcm_decoder4 = ResidualBlock(512, 256)
        self.lcm_upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.lcm_decoder3 = ResidualBlock(256, 128)
        self.lcm_upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.lcm_decoder2 = ResidualBlock(128, 64)
        self.lcm_upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.lcm_decoder1 = ResidualBlock(64, 32)
        self.lcm_conv_out = nn.Conv2d(32, num_lcm_classes, kernel_size=1)
        
        # Change detection branch
        # Takes concatenated features from both images + difference of LCM features
        self.cd_bottleneck = ResidualBlock(512*2, 512)
        
        self.cd_upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.cd_decoder4 = ResidualBlock(256*3, 256)  # 256 from upconv + 256 from each image
        self.cd_upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.cd_decoder3 = ResidualBlock(128*3, 128)
        self.cd_upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.cd_decoder2 = ResidualBlock(64*3, 64)
        self.cd_upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.cd_decoder1 = ResidualBlock(32*3, 32)
        self.cd_conv_out = nn.Conv2d(32, num_cd_classes, kernel_size=1)
    
    def forward_lcm_branch(self, x):
        """Forward pass for the land cover mapping branch"""
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Bottleneck
        bottleneck = self.lcm_bottleneck(enc4)
        
        # Decoder with skip connections
        dec4 = self.lcm_upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.lcm_decoder4(dec4)
        
        dec3 = self.lcm_upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.lcm_decoder3(dec3)
        
        dec2 = self.lcm_upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.lcm_decoder2(dec2)
        
        dec1 = self.lcm_upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.lcm_decoder1(dec1)
        
        # Output
        lcm_out = self.lcm_conv_out(dec1)
        return lcm_out, [dec4, dec3, dec2, dec1]
    
    def forward(self, x1, x2):
        # Forward pass for both images through LCM branches
        lcm1_out, lcm1_features = self.forward_lcm_branch(x1)
        lcm2_out, lcm2_features = self.forward_lcm_branch(x2)
        
        # Get encoder features for both images (for CD branch)
        with torch.no_grad():
            enc1_1 = self.encoder1(x1)
            enc2_1 = self.encoder2(enc1_1)
            enc3_1 = self.encoder3(enc2_1)
            enc4_1 = self.encoder4(enc3_1)
            bottleneck_1 = self.lcm_bottleneck(enc4_1)
            
            enc1_2 = self.encoder1(x2)
            enc2_2 = self.encoder2(enc1_2)
            enc3_2 = self.encoder3(enc2_2)
            enc4_2 = self.encoder4(enc3_2)
            bottleneck_2 = self.lcm_bottleneck(enc4_2)
        
        # Change detection branch
        # Concatenate features from both images
        cd_features = torch.cat([bottleneck_1, bottleneck_2], dim=1)
        cd_out = self.cd_bottleneck(cd_features)
        
        # Decoder with skip connections from both images and difference features
        cd_dec4 = self.cd_upconv4(cd_out)
        # Difference of LCM features at this level
        lcm_diff4 = torch.abs(lcm1_features[0] - lcm2_features[0])
        cd_dec4 = torch.cat([cd_dec4, enc4_1, enc4_2, lcm_diff4], dim=1)
        cd_dec4 = self.cd_decoder4(cd_dec4)
        
        cd_dec3 = self.cd_upconv3(cd_dec4)
        lcm_diff3 = torch.abs(lcm1_features[1] - lcm2_features[1])
        cd_dec3 = torch.cat([cd_dec3, enc3_1, enc3_2, lcm_diff3], dim=1)
        cd_dec3 = self.cd_decoder3(cd_dec3)
        
        cd_dec2 = self.cd_upconv2(cd_dec3)
        lcm_diff2 = torch.abs(lcm1_features[2] - lcm2_features[2])
        cd_dec2 = torch.cat([cd_dec2, enc2_1, enc2_2, lcm_diff2], dim=1)
        cd_dec2 = self.cd_decoder2(cd_dec2)
        
        cd_dec1 = self.cd_upconv1(cd_dec2)
        lcm_diff1 = torch.abs(lcm1_features[3] - lcm2_features[3])
        cd_dec1 = torch.cat([cd_dec1, enc1_1, enc1_2, lcm_diff1], dim=1)
        cd_dec1 = self.cd_decoder1(cd_dec1)
        
        # Change detection output
        cd_out = self.cd_conv_out(cd_dec1)
        
        return lcm1_out, lcm2_out, cd_out

class SequentialTrainer:
    """Implements the sequential training scheme from the paper"""
    def __init__(self, model, lcm_criterion, cd_criterion, lcm_optimizer, cd_optimizer, device):
        self.model = model.to(device)
        self.lcm_criterion = lcm_criterion
        self.cd_criterion = cd_criterion
        self.lcm_optimizer = lcm_optimizer
        self.cd_optimizer = cd_optimizer
        self.device = device
    
    def train_lcm_phase(self, train_loader, num_epochs):
        """Train only the LCM branches"""
        self.model.train()
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (x1, x2, lcm1, lcm2, cd) in enumerate(train_loader):
                x1, x2 = x1.to(self.device), x2.to(self.device)
                lcm1, lcm2 = lcm1.to(self.device), lcm2.to(self.device)
                
                # Zero gradients
                self.lcm_optimizer.zero_grad()
                
                # Forward pass (only LCM branches)
                lcm1_pred, _ = self.model.forward_lcm_branch(x1)
                lcm2_pred, _ = self.model.forward_lcm_branch(x2)
                
                # Compute loss
                loss = self.lcm_criterion(lcm1_pred, lcm1) + self.lcm_criterion(lcm2_pred, lcm2)
                
                # Backward pass and optimize
                loss.backward()
                self.lcm_optimizer.step()
                
                running_loss += loss.item()
            
            print(f"LCM Phase Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    def train_cd_phase(self, train_loader, num_epochs):
        """Train the CD branch while keeping LCM branches fixed"""
        self.model.train()
        
        # Freeze LCM branch parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze CD branch parameters
        for param in self.model.cd_bottleneck.parameters():
            param.requires_grad = True
        for param in self.model.cd_upconv4.parameters():
            param.requires_grad = True
        for param in self.model.cd_decoder4.parameters():
            param.requires_grad = True
        for param in self.model.cd_upconv3.parameters():
            param.requires_grad = True
        for param in self.model.cd_decoder3.parameters():
            param.requires_grad = True
        for param in self.model.cd_upconv2.parameters():
            param.requires_grad = True
        for param in self.model.cd_decoder2.parameters():
            param.requires_grad = True
        for param in self.model.cd_upconv1.parameters():
            param.requires_grad = True
        for param in self.model.cd_decoder1.parameters():
            param.requires_grad = True
        for param in self.model.cd_conv_out.parameters():
            param.requires_grad = True
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (x1, x2, lcm1, lcm2, cd) in enumerate(train_loader):
                x1, x2 = x1.to(self.device), x2.to(self.device)
                cd = cd.to(self.device)
                
                # Zero gradients
                self.cd_optimizer.zero_grad()
                
                # Forward pass (full model but only CD output is used)
                _, _, cd_pred = self.model(x1, x2)
                
                # Compute loss
                loss = self.cd_criterion(cd_pred, cd)
                
                # Backward pass and optimize
                loss.backward()
                self.cd_optimizer.step()
                
                running_loss += loss.item()
            
            print(f"CD Phase Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    def validate(self, val_loader):
        """Validate both LCM and CD performance"""
        self.model.eval()
        lcm_loss = 0.0
        cd_loss = 0.0
        
        with torch.no_grad():
            for x1, x2, lcm1, lcm2, cd in val_loader:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                lcm1, lcm2 = lcm1.to(self.device), lcm2.to(self.device)
                cd = cd.to(self.device)
                
                # Forward pass
                lcm1_pred, lcm2_pred, cd_pred = self.model(x1, x2)
                
                # Compute losses
                lcm_loss += (self.lcm_criterion(lcm1_pred, lcm1) + self.lcm_criterion(lcm2_pred, lcm2)).item()
                cd_loss += self.cd_criterion(cd_pred, cd).item()
        
        avg_lcm_loss = lcm_loss / (2 * len(val_loader))  # Divide by 2 for two LCM outputs
        avg_cd_loss = cd_loss / len(val_loader)
        
        print(f"Validation - LCM Loss: {avg_lcm_loss:.4f}, CD Loss: {avg_cd_loss:.4f}")
        return avg_lcm_loss, avg_cd_loss

class ChangeMask(nn.Module):
    """Complete ChangeMask model"""
    def __init__(self, num_classes=11):
        super().__init__()
        # Shared semantic-aware encoder (Siamese structure)
        self.encoder = nn.Sequential(
            nn.Conv2d(138, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Temporal Symmetric Transformer (TST)
        self.tst = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Multi-task decoder
        self.decoder_seg = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        
        self.decoder_cd = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1)
        )
        
    def forward(self, x1, x2):
        # Encoding phase
        feats1 = self.encoder(x1)  # Multi-scale features for image 1
        feats2 = self.encoder(x2)  # Multi-scale features for image 2
        
        # TST processing
        z1, z2 = feats1, feats2
        z_fused = self.tst(z1 + z2)  # Simple fusion instead of complex TST
        
        # Semantic segmentation branch (for both time phases)
        seg_out1 = self.decoder_seg(z1)  # Time phase 1
        seg_out2 = self.decoder_seg(z2)  # Time phase 2
        
        # Change detection branch (fusing dual-temporal features)
        cd_out = self.decoder_cd(z_fused)
        
        # Global average pooling to get final outputs
        seg_out1 = F.adaptive_avg_pool2d(seg_out1, 1).squeeze(-1).squeeze(-1)  # [B, num_classes]
        seg_out2 = F.adaptive_avg_pool2d(seg_out2, 1).squeeze(-1).squeeze(-1)  # [B, num_classes]
        cd_out = F.adaptive_avg_pool2d(cd_out, 1).squeeze(-1).squeeze(-1)      # [B, 2]
        

        return {
            'change': cd_out,      # [B, 2]
            'class_2010': seg_out1,      # [B, num_classes]
            'class_2015': seg_out2       # [B, num_classes]
        }

if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    
    # Create model instance
    model = ChangeMask(num_classes=11)
    
    # Create test inputs
    batch_size = 2
    x1 = torch.randn(batch_size, 138, 13, 13)  # Phase 1 image
    x2 = torch.randn(batch_size, 138, 13, 13)  # Phase 2 image
    
    # Forward pass
    try:
        outputs = model(x1, x2)
        print("Model test successful!")
        print(f"Input shapes: {x1.shape}, {x2.shape}")
        print(f"Change detection output shape: {outputs['change'].shape}")       # [batch_size, 2]
        print(f"Semantic segmentation output shape (T1): {outputs['class_2010'].shape}")  # [batch_size, num_classes]
        print(f"Semantic segmentation output shape (T2): {outputs['class_2015'].shape}")  # [batch_size, num_classes]
    except Exception as e:
        print(f"Model test failed: {str(e)}")
        
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

