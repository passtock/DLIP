import torch
import torch.nn as nn
import timm

class RegionEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=128):
        super().__init__()
        base = timm.create_model('efficientnet_b2', pretrained=False)
        old = base.conv_stem
        base.conv_stem = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        base.classifier = nn.Sequential(nn.Linear(base.classifier.in_features, out_dim), nn.ReLU())
        self.encoder = base
    def forward(self, x): return self.encoder(x)

class TransformerFusion(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, depth=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*2, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        
    def forward(self, x):
        # 🎯 핵심 해결책: 트랜스포머를 통과할 때만 AMP(16비트)를 강제로 끄고 32비트(Float)로 실행!
        with torch.cuda.amp.autocast(enabled=False):
            x_f = x.float()
            cls_tokens = self.cls_token.expand(x_f.size(0), -1, -1).float()
            out = self.transformer(torch.cat((cls_tokens, x_f), dim=1))[:, 0]
        return out

class PSAMultiBranchModel(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.f_enc = RegionEncoder(6, embed_dim)
        self.c_enc = RegionEncoder(24, embed_dim)
        self.e_enc = RegionEncoder(24, embed_dim)
        self.s_enc = RegionEncoder(6, embed_dim)
        self.fusion = TransformerFusion(embed_dim=embed_dim)
        self.clf = nn.Sequential(nn.Dropout(0.2), nn.Linear(embed_dim, 2))
        
    def forward(self, f, c, e, s):
        features = torch.stack([self.f_enc(f), self.c_enc(c), self.e_enc(e), self.s_enc(s)], dim=1)
        return self.clf(self.fusion(features))
