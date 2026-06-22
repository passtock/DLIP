import torch
import torch.nn as nn
import timm

class CustomEncoder(nn.Module):
    def __init__(self, in_ch, out_dim):
        super().__init__()
        # 가벼운 ResNet18을 4개의 뇌에 각각 탑재
        self.bb = timm.create_model('resnet18', pretrained=False, num_classes=out_dim)
        # 입력 채널(6, 24 등)에 맞게 첫 번째 Conv 레이어 개조
        self.bb.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    def forward(self, x): return self.bb(x)

class PSAMultiBranchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = CustomEncoder(6, 128)
        self.c = CustomEncoder(24, 128)
        self.e = CustomEncoder(24, 128)
        self.s = CustomEncoder(6, 128)
        self.clf = nn.Sequential(nn.Linear(128*4, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))
        
    def forward(self, f, c, e, s):
        # 4개의 부위를 분석한 결과를 하나로 합쳐서 최종 등급 판정
        return self.clf(torch.cat([self.f(f), self.c(c), self.e(e), self.s(s)], dim=1))
