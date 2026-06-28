import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout2d(0.5) if dropout else nn.Identity()

    def forward(self, x, skip_x):
        x = self.act(self.bn(self.upconv(x)))
        x = self.drop(x)
        
        diffY = skip_x.size()[2] - x.size()[2]
        diffX = skip_x.size()[3] - x.size()[3]
        
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        return torch.cat([x, skip_x], dim=1)

class AudioUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = ConvBlock(1, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        
      
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
       
        self.dec3 = UpConvBlock(128, 64, dropout=True)
        self.dec2 = UpConvBlock(128, 32) 
        self.dec1 = UpConvBlock(64, 16)
        
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
    
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        b = self.bottleneck(e3)
        
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        
        mask = self.final_conv(d1)
        
        denoised_mag = x * mask
        return denoised_mag