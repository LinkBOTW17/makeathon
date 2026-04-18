import torch
import torch.nn as nn
import torch.nn.functional as F

class StatisticalTemporalEncoder(nn.Module):
    """
    Upgraded Temporal Encoder replacing the naive UTAEPlaceholder.
    By concatenating the Min, Max, and Mean across time, the model can 
    natively measure variance and spot distinct chronological anomalies 
    (like a pristine forest turning suddenly barren in month 7).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Input to conv is now 3x the channels because we concatenate Min, Max, Mean!
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        x_min = x.min(dim=1)[0]
        x_max = x.max(dim=1)[0]
        x_mean = x.mean(dim=1)
        
        # Concat across temporal statistics -> (B, C*3, H, W)
        x_stat = torch.cat([x_min, x_max, x_mean], dim=1)
        return self.conv(x_stat)

class FusionNet(nn.Module):
    def __init__(self, s1_channels=2, s2_channels=10, aef_channels=768, num_classes=1):
        super().__init__()
        
        # S1 branch (Radar - mostly 2 bands: VV, VH)
        self.s1_encoder = nn.Sequential(
            nn.Conv3d(s1_channels, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, None, None)) # Pool over time
        )
        self.s1_proj = nn.Conv2d(16, 64, kernel_size=1)
        
        # S2 branch (Optical - e.g. 12 bands like B2, B3, B4, B8, etc)
        self.s2_encoder = StatisticalTemporalEncoder(s2_channels, 64)
        
        # AEF Foundation Model Embeddings branch
        self.aef_proj = nn.Sequential(
            nn.Conv2d(aef_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Decoder
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(64 + 64 + 128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3), # For MC Dropout Uncertainty Estimation
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        
    def forward(self, s1, s2, aef):
        """
        s1: (B, T, C_s1, H, W)
        s2: (B, T, C_s2, H, W)
        aef: (B, C_aef, H_aef, W_aef) Note: might need upsampling if AEF res differs
        """
        B = s2.shape[0]
        
        # Process S1
        # PyTorch Conv3d expects (B, C, T, H, W)
        if s1.dim() == 5:
            s1 = s1.permute(0, 2, 1, 3, 4)
            s1_feat = self.s1_encoder(s1)          # (B, 16, 1, H, W)
            s1_feat = s1_feat.squeeze(2)             # (B, 16, H, W)
            s1_feat = self.s1_proj(s1_feat)          # (B, 64, H, W)
        else:
            H, W = s2.shape[-2], s2.shape[-1]
            s1_feat = torch.zeros(B, 64, H, W, device=s2.device)
            
        # Process S2
        if s2.dim() == 5:
            s2_feat = self.s2_encoder(s2)            # (B, 64, H, W)
        else:
            H, W = s1.shape[-2], s1.shape[-1]
            s2_feat = torch.zeros(B, 64, H, W, device=s1.device)
            
        # Process AEF embeddings
        if aef.dim() == 4:
            aef_feat = self.aef_proj(aef)            # (B, 128, H_aef, W_aef)
            H, W = s2_feat.shape[-2], s2_feat.shape[-1]
            if aef_feat.shape[-2:] != (H, W):
                aef_feat = F.interpolate(aef_feat, size=(H, W), mode='bilinear', align_corners=False)
        else:
            H, W = s2_feat.shape[-2], s2_feat.shape[-1]
            aef_feat = torch.zeros(B, 128, H, W, device=s2.device)
            
        # Concat and decode
        fused = torch.cat([s1_feat, s2_feat, aef_feat], dim=1) # (B, 256, H, W)
        logits = self.fusion_conv(fused)                     # (B, num_classes, H, W)
        
        return logits
